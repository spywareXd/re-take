"""
retake_keeper.py: run the ReTake keeper decision model without the ReTake repository.

    pip install torch transformers safetensors huggingface_hub bitsandbytes        # bitsandbytes is only for the default 4-bit mode (CUDA GPU)

    from retake_keeper import RetakeKeeper
    keeper = RetakeKeeper.from_pretrained("spywarex/retake-keeper-qwen3-4b")        # downloads about 3.5 GB once
    r = keeper.decide([{"start": 12.0, "end": 14.1, "text": "So the main idea is"},
                       {"start": 15.0, "end": 18.0, "text": "So the main idea is that it works."}])
    print(r["keep"], r["unclear"], r["decided_by"])

The model is Qwen3-4B-Instruct-2507 (QLoRA) with Cloudflare's Clef joint decision head. Given a group of repeated takes of one line it returns p(keep) for every
take; decide() adds the ReTake rules around that (cue phrases keep none, identical takes keep the last, an empty answer keeps the last, unsure answers keep the
last and are flagged). Record encoding and head follow Cloudflare's joint_schema_model.py (Apache-2.0, https://huggingface.co/Cloudflare/clef).
Modes: "4bit" (default; the trained pre-quantized base plus LoRA in 16 bit; about 5 GB of VRAM) and "fp16" (merged weights; about 8 GB, so it needs a 10 GB GPU
or it offloads to the CPU and is slow). Compute is bf16 autocast, as in training.
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MODES = ("4bit", "fp16")
SERVE_MAX_LENGTH = 16384                       # Unsloth's predict() reads up to 16384 tokens, whatever the training length
MIN_TRANSFORMERS = (4, 51)                     # first release with Qwen3

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


class DecisionModelError(RuntimeError):
    """Base class of every error this module raises on purpose."""


class DecisionModelMissing(DecisionModelError, FileNotFoundError):
    """The model folder (or a file in it) is not there."""


class DecisionModelIncompatible(DecisionModelError):
    """A library version, a missing dependency or a file that does not match the head."""


# --------------------------------------------------------------------------------------------------------------------- record encoding
def parse_state(value: Any) -> Any:
    """As FastDecisionModel.predict does: a string that looks like JSON becomes the object; anything else is left alone."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_question(question: Any) -> dict:
    """Only yes/no (noul) questions are scored here: {"type": "noul", "instructions": ...}."""
    if not isinstance(question, dict) or question.get("type") != "noul":
        raise ValueError("only questions of type 'noul' are supported")
    criteria = question.get("criteria")
    if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
        raise ValueError('criteria may only have "true" and "false"')
    out = {"type": "noul", "instructions": question.get("instructions")}
    if criteria is not None:
        out["criteria"] = criteria
    return out


def question_options(question: dict) -> list[tuple[str, Any]]:
    criteria = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
    criteria.update(question.get("criteria") or {})
    return [(key, criteria[key]) for key in ("true", "false")]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_record(tokenizer: Any, state: Any, questions: dict, max_length: int = SERVE_MAX_LENGTH) -> EncodedRecord:
    """Token ids of the chat-templated prompt (system, STATE, SCHEMA FIELDS with one block per question) and the token spans the head reads."""
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    encoded: list[EncodedQuestion] = []
    for index, (question_id, question) in enumerate(questions.items()):
        schema_ids.extend(_tokens(tokenizer, f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: "))
        start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        spans, ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            spans.append((option_start, len(schema_ids)))
            ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        encoded.append(EncodedQuestion(str(question_id), (start, end), tuple(spans), tuple(ids)))
    prefix = _tokens(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = _tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    state_ids = _tokens(tokenizer, render(state))
    fixed = len(prefix) + len(schema_ids) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed]                      # a state that does not fit is cut at the end
    offset = len(prefix) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(q.question_id, (q.question_span[0] + offset, q.question_span[1] + offset),
                        tuple((a + offset, b + offset) for a, b in q.option_spans), q.option_ids)
        for q in encoded
    )
    if not shifted:
        raise ValueError("at least one question is required")
    return EncodedRecord(tuple(prefix + state_ids + schema_ids + suffix), shifted)


# --------------------------------------------------------------------------------------------------------------------- the head
def _build_head_classes():
    import torch
    import torch.nn.functional as functional

    class EvidenceRoutingLayer(torch.nn.Module):
        def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.query_norm = torch.nn.LayerNorm(width)
            self.memory_norm = torch.nn.LayerNorm(width)
            self.attention = torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.attention_dropout = torch.nn.Dropout(dropout)
            self.feedforward_norm = torch.nn.LayerNorm(width)
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(feedforward, width), torch.nn.Dropout(dropout))

        def forward(self, queries, memory):
            routed, _ = self.attention(self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
            queries = queries + self.attention_dropout(routed)
            return queries + self.feedforward(self.feedforward_norm(queries))

    class JointSchemaHead(torch.nn.Module):
        def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.hidden_norm = torch.nn.LayerNorm(hidden_size)
            self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
            self.type_embedding = torch.nn.Embedding(3, width)
            self.evidence_layers = torch.nn.ModuleList([EvidenceRoutingLayer(width, heads, feedforward, dropout) for _ in range(routing_layers)])
            self.option_summary_norm = torch.nn.LayerNorm(width)
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerDecoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
                for _ in range(layers)])
            self.field_norm = torch.nn.LayerNorm(width)
            self.option_norm = torch.nn.LayerNorm(width)
            self.residual_scorer = torch.nn.Sequential(torch.nn.Linear(width * 4, width), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(width, 1))
            self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
            self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
            self.residual_gate = torch.nn.Parameter(torch.zeros(()))

        def forward(self, hidden_states, input_ids, record: EncodedRecord, output_embedding_weight):
            """hidden_states (1, T, hidden), input_ids (1, T) -> one logits tensor (options,) per question. Batch size 1: no padding."""
            sequence_hidden = self.hidden_norm(hidden_states)[0]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            dtype = self.hidden_norm.weight.dtype
            question_vectors = torch.stack([sequence_hidden[a:b].mean(dim=0) for a, b in (q.question_span for q in record.questions)])
            type_ids = torch.zeros(len(record.questions), dtype=torch.long, device=hidden_states.device)           # every question is a noul (type 0)
            option_counts = [len(q.option_spans) for q in record.questions]
            contexts, lexicals = [], []
            for q in record.questions:
                contexts.append(torch.stack([sequence_hidden[a:b].mean(dim=0) for a, b in q.option_spans]))
                lexicals.append(torch.stack([output_embedding_weight[input_ids[0, a:b]].to(dtype).mean(dim=0) for a, b in q.option_spans]))
            queries = [self.option_context_projection(c) + self.option_lexical_projection(l) + self.option_question_projection(question_vectors[i]).unsqueeze(0)
                       for i, (c, l) in enumerate(zip(contexts, lexicals))]
            routed = torch.cat(queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            split = list(torch.split(routed[0], option_counts, dim=0))
            base_fields = self.question_projection(question_vectors)
            summaries = []
            for field, options in zip(base_fields, split):
                weights = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
                summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim=0))
            fields = base_fields + self.option_summary_norm(torch.stack(summaries)) + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(type_ids)
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])
            logits = []
            prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
            joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
            for i, (field, lexical, routed_options) in enumerate(zip(fields, lexicals, split)):
                anchor = functional.normalize(question_vectors[i] + global_vector, dim=-1)
                prior = prior_scale * torch.matmul(functional.normalize(lexical, dim=-1), anchor)
                options = self.option_norm(routed_options)
                repeated = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated, options, dim=-1)
                features = torch.cat([repeated, options, repeated * options, torch.abs(repeated - options)], dim=-1)
                residual = self.residual_scorer(features).squeeze(-1)
                logits.append(prior + torch.sigmoid(self.residual_gate) * (joint_scale * cosine + residual))
            return logits

    return JointSchemaHead


# --------------------------------------------------------------------------------------------------------------------- the model
def _version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for part in text.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _register_causal_attention() -> str:
    """Attention for one unpadded sequence: expand the KV heads and call SDPA with is_causal=True (no T x T mask, so the memory-efficient kernel is used and 16k tokens fit in 8 GB)."""
    import torch
    from transformers import AttentionInterface

    def causal_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **_kwargs):
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=scaling, is_causal=True)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("standalone_causal", causal_sdpa)
    return "standalone_causal"


class StandaloneDecisionModel:
    """Lazy loader and scorer. `score(state, questions)` returns {question id: p_yes}; `unload()` frees the VRAM."""

    def __init__(self, model_dir: str | Path, mode: str = "4bit", device: str = "cuda", max_length: int = SERVE_MAX_LENGTH,
                 gpu_memory_gb: Optional[float] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model_dir = Path(model_dir)
        self.mode, self.device, self.max_length, self.gpu_memory_gb = mode, device, int(max_length), gpu_memory_gb
        self._lm = self._head = self._tokenizer = self._hooks = None
        self.decision_config: dict = {}
        self.load_seconds: Optional[float] = None

    # ----------------------------------------------------------------------------------------------------------- availability and loading
    def required_files(self) -> list[Path]:
        d = self.model_dir
        common = [d / "head.safetensors", d / "head_config.json", d / "tokenizer.json", d / "decision_config.json"]
        if self.mode == "4bit":
            return common + [d / "base_bnb4" / "model.safetensors", d / "base_bnb4" / "config.json", d / "adapter" / "adapter_model.safetensors", d / "adapter" / "adapter_config.json"]
        return common + [d / "config.json", d / "model.safetensors.index.json"]

    def availability(self) -> tuple[bool, str]:
        """(usable, reason) without loading anything: files, libraries and a CUDA device."""
        missing = [str(p.relative_to(self.model_dir)) for p in self.required_files() if not p.is_file()]
        if not self.model_dir.is_dir() or missing:
            return False, f"model folder {self.model_dir} is missing: {', '.join(missing) or 'the folder itself'}"
        try:
            import torch
            import transformers
        except ImportError as exc:
            return False, f"torch and transformers are required ({exc})"
        if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
            return False, f"transformers {transformers.__version__} is too old for Qwen3 (need >= {'.'.join(map(str, MIN_TRANSFORMERS))})"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return False, "no CUDA device is available"
        if self.mode == "4bit":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                return False, f"mode '{self.mode}' needs bitsandbytes"
        return True, "ok"

    def _load(self) -> None:
        ok, why = self.availability()
        if not ok:
            raise (DecisionModelMissing if "missing" in why else DecisionModelIncompatible)(why)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        attn = _register_causal_attention()
        d = self.model_dir
        self.decision_config = json.loads((d / "decision_config.json").read_text(encoding="utf-8"))
        # temperatures: the fitted head temperature is folded into the head's weights; what is left to apply is 1.0 unless the config says otherwise
        self._temperature = float(self.decision_config.get("head_temperature", 1.0)) * float((self.decision_config.get("temperature") or [1.0, 1.0, 1.0])[2])
        self._tokenizer = AutoTokenizer.from_pretrained(str(d))
        dev = torch.device(self.device)
        if self.mode == "4bit":
            self._lm = AutoModelForCausalLM.from_pretrained(str(d / "base_bnb4"), device_map={"": dev.index if dev.index is not None else 0}, dtype=torch.bfloat16, attn_implementation=attn)
            self._attach_lora(d / "adapter")
        elif self.mode == "fp16":
            kwargs: dict[str, Any] = {"dtype": torch.float16, "attn_implementation": attn}
            free = torch.cuda.mem_get_info(dev)[0] / 2**30 if dev.type == "cuda" else 0.0
            if free >= 9.0 and self.gpu_memory_gb is None:
                kwargs["device_map"] = {"": dev.index if dev.index is not None else 0}
            else:                                                       # the weights do not fit: keep the layers that fit on the GPU, offload the rest to the CPU
                limit = self.gpu_memory_gb if self.gpu_memory_gb is not None else max(free - 1.5, 1.0)
                kwargs.update(device_map="auto", max_memory={0: f"{limit:.1f}GiB", "cpu": "30GiB"})
            self._lm = AutoModelForCausalLM.from_pretrained(str(d), **kwargs)
        self._lm.config.use_cache = False
        self._lm.eval()
        head_config = json.loads((d / "head_config.json").read_text(encoding="utf-8"))
        Head = _build_head_classes()
        head = Head(**head_config)
        try:
            head.load_state_dict(load_file(str(d / "head.safetensors")), strict=True)
        except RuntimeError as exc:
            raise DecisionModelIncompatible(f"head.safetensors does not match head_config.json: {str(exc)[:300]}") from exc
        if head_config["hidden_size"] != self._lm.config.hidden_size:
            raise DecisionModelIncompatible(f"head hidden_size {head_config['hidden_size']} != backbone {self._lm.config.hidden_size}")
        self._head = head.to(device=dev, dtype=torch.float32).eval()
        self.load_seconds = time.time() - t0

    def _attach_lora(self, adapter_dir: Path) -> None:
        """LoRA applied as forward hooks on the 4-bit base: out + scaling * (x @ A^T) @ B^T, in the compute dtype, exactly what PEFT does under autocast."""
        import re

        import torch
        from safetensors import safe_open

        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        scaling = float(cfg["lora_alpha"]) / (float(cfg["r"]) ** 0.5 if cfg.get("use_rslora") else float(cfg["r"]))
        pairs: dict[str, dict[str, Any]] = {}
        dev = next(self._lm.parameters()).device
        with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
            for key in f.keys():
                m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", key)
                if not m:
                    raise DecisionModelIncompatible(f"unexpected tensor {key} in the adapter")
                pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(key).to(dev, torch.bfloat16)
        self._hooks = []
        for name, ab in pairs.items():
            try:
                module = self._lm.get_submodule(name)
            except AttributeError as exc:
                raise DecisionModelIncompatible(f"adapter targets {name}, which the base model does not have") from exc
            a, b = ab["A"], ab["B"]

            def hook(_module, args, output, a=a, b=b):
                x = args[0].to(a.dtype)
                return output + ((x @ a.t()) @ b.t()).to(output.dtype) * scaling

            self._hooks.append(module.register_forward_hook(hook))

    @property
    def loaded(self) -> bool:
        return self._lm is not None

    # ----------------------------------------------------------------------------------------------------------- scoring
    def score_logits(self, state: Any, questions: dict) -> tuple[dict[str, list[float]], int]:
        """({question id: [logit true, logit false]}, number of input tokens)."""
        import torch

        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty dict of name to question")
        questions = {str(k): normalize_question(v) for k, v in questions.items()}
        if not self.loaded:
            self._load()
        record = encode_record(self._tokenizer, parse_state(state), questions, self.max_length)
        dev = next(self._head.parameters()).device
        input_ids = torch.tensor([record.input_ids], dtype=torch.long)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            first = next(self._lm.parameters()).device
            # batch of one, no padding: no attention mask, so SDPA takes its causal memory-efficient path instead of materializing a T x T mask
            hidden = self._lm.model(input_ids=input_ids.to(first), use_cache=False, return_dict=True).last_hidden_state
            embedding = self._lm.get_output_embeddings().weight
            logits = self._head(hidden.to(dev, torch.bfloat16), input_ids.to(dev), record, embedding.to(dev) if embedding.device != dev else embedding)
        out = {q.question_id: [float(v) for v in z.float().cpu().tolist()] for q, z in zip(record.questions, logits)}
        return out, len(record.input_ids)

    def score(self, state: Any, questions: dict) -> dict[str, float]:
        """{question id: probability of yes} (unrounded), the softmax over [true, false] at the served temperature."""
        logits, _ = self.score_logits(state, questions)
        out = {}
        for qid, z in logits.items():
            z = [v / self._temperature for v in z]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            out[qid] = e[0] / sum(e)                                    # option order is (true, false)
        return out

    def unload(self) -> None:
        """Free the model, the head and the GPU memory. The next score() loads again."""
        if self._hooks:
            for h in self._hooks:
                h.remove()
        self._lm = self._head = self._tokenizer = self._hooks = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> "StandaloneDecisionModel":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()


# ================================================================================================================ rules, input builders, policy
# Everything below turns a list of takes into the model's input and applies the ReTake hybrid keeper policy. It is a copy of the training-time builders
# (training/build_decision_data*.py in the ReTake repository) so that the model sees exactly the format it was trained on.
import hashlib  # noqa: E402
import re  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402

RULES = 'Choose which takes of a repeated line stay; the rest are deleted.\nAnswer each yes/no question: should this take stay in the final video?\nUse only the given ids, never a [context] unit. Keep at least one id unless keep_none.\n1. Drop truncated, stumbling or abandoned takes.\n2. Drop takes with a mistake a later take fixes.\n3. Of complete takes keep the LATER (a restart within ~1 s means the earlier failed); identical: the last.\n4. Later take drops key content: keep the earlier, unclear.\n5. Next unit continues the sentence (gap <= ~1 s, no repeated words): keep both.\n6. Prefer the take that fits the context.\n7. Cue phrases ("let\'s go", "here we go"): keep [], keep_none true.\n8. Two lines mixed: keep the last of each, unclear.\n9. Text only; paraphrases: keep the later, unclear.\nBe conservative: never drop the only good take.'
CUES = ["Okay, here we go.", "Alright, from the top.", "Let me start again.", "Take two.", "Okay, rolling.", "Here we go again.", "Let's go.", "Ready, from the beginning.", "One more time.", "Okay, again.", "Take three.", "Sorry, let me redo that.", "And action.", "Okay, let's try that again.", "From the top, one more.", "Starting over.", "Okay, take four.", "Let me do that once more.", "Alright, new take.", "Here we go, take five.", "Okay, going again.", "Reset, and again.", "Alright, rolling now.", "Okay, one more time from the start.", "Ready, here we go.", "Take six, from the top.", "Let's go.", "I'm gonna start from the top.", "Here we go.", "let me start from the top."]
CONTEXT_MARK = "[context, not in this group]"
MAX_WORDS = 40
THRESHOLD, UNCLEAR_MARGIN, UNCLEAR_MAX_BELOW = 0.5, 0.15, 0.6
CUE_THRESHOLD, CUE_MAX_WORDS = 0.8, 8


def _f2(x: float) -> float:
    return float(f"{float(x):.2f}")


def _unit(t: dict, i: int = 0) -> dict:
    return {"id": str(t.get("id", t.get("unit_id", i + 1))), "start": float(t["start"]), "end": float(t["end"]), "text": str(t.get("text", t.get("raw_text", "")))}


def build_state(takes: list[dict], before: Optional[dict] = None, after: Optional[dict] = None) -> dict:
    """takes: dicts with start, end (seconds), text and an optional id (default 1, 2, ...). before / after: the sentence just before / after the group (same keys) or None."""
    units = sorted((_unit(t, i) for i, t in enumerate(takes)), key=lambda u: (u["start"], u["end"], u["id"]))
    if not units:
        raise ValueError("at least one take is required")
    b = _unit(before) if before else None
    a = _unit(after) if after else None
    prev_end, out = (b["end"] if b else None), []
    for u in units:
        gap = None if prev_end is None else _f2(max(0.0, u["start"] - prev_end))
        out.append({"id": u["id"], "start": _f2(u["start"]), "end": _f2(u["end"]), "gap_before_s": gap, "text": u["text"]})
        prev_end = u["end"]
    ctx = {side: (None if c is None else {"id": c["id"], "start": _f2(c["start"]), "end": _f2(c["end"]), "text": c["text"], "note": CONTEXT_MARK}) for side, c in (("before", b), ("after", a))}
    return {"rules": RULES, "takes": out, "context": ctx}


def _truncate_words(text: str) -> str:
    words = text.split()
    return text.strip() if len(words) <= MAX_WORDS else " ".join(words[:MAX_WORDS]) + "…"


def build_questions(state: dict) -> dict:
    n = len(state["takes"])
    return {f'keep_{t["id"]}': {"type": "noul", "instructions": f'Should take {t["id"]} ("{_truncate_words(t["text"])}", take {i + 1} of {n}) stay in the final video?'}
            for i, t in enumerate(state["takes"])}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']+|_", " ", str(text).lower())).strip()


_CUE_SET = {normalise(c) for c in CUES}


def _cue_score(text: str) -> float:
    n = normalise(text)
    if n in _CUE_SET:
        return 1.0
    w, best = set(n.split()), 0.0
    for c in _CUE_SET:
        cw = set(c.split())
        best = max(best, SequenceMatcher(None, n, c).ratio(), len(w & cw) / max(len(w | cw), 1))
    return best


def _is_cue_take(text: str) -> bool:
    return 0 < len(normalise(text).split()) <= CUE_MAX_WORDS and _cue_score(text) >= CUE_THRESHOLD


def _decide_probs(probs: dict[str, float]) -> dict:
    keep = [u for u, v in probs.items() if v >= THRESHOLD]
    ranked = sorted(probs.values(), reverse=True)
    unclear = bool(ranked) and (ranked[0] < UNCLEAR_MAX_BELOW or (len(ranked) > 1 and ranked[0] - ranked[1] <= UNCLEAR_MARGIN))
    return {"keep": keep, "unclear": unclear if ranked else True}


def decide_group(takes: list[dict], before: Optional[dict], after: Optional[dict], predict, cue_rule: bool = True, exact_gate: bool = True,
                 empty_fallback: bool = True, unclear_default: bool = True) -> dict:
    """The ReTake hybrid policy for one group. `predict(state, questions) -> {"keep_<id>": p_yes}`.
    Layers in order: cue rule (every take is a short production cue -> keep none), exact gate (every take has the same words -> keep the last), the model
    (keep = p >= 0.5), empty fallback (model keeps nothing -> keep the last), unclear (top two within 0.15 or max p < 0.6 -> keep the last, flagged).
    Returns {"keep": [ids], "keep_none": bool, "unclear": bool, "decided_by": str, "reason": str, "probs": {id: p} | None}."""
    state = build_state(takes, before, after)
    ids = [t["id"] for t in state["takes"]]
    texts = [t["text"] for t in state["takes"]]

    def result(keep, unclear, by, reason, probs=None, **extra):
        return {"keep": list(keep), "keep_none": not keep, "unclear": bool(unclear), "decided_by": by, "reason": reason, "probs": probs, **extra}

    if cue_rule and all(_is_cue_take(t) for t in texts):
        return result([], False, "cue_rule", "Every take is a production cue: keep none.")
    norm = [normalise(t) for t in texts]
    if exact_gate and len(norm) >= 2 and norm[0] and all(n == norm[0] for n in norm):
        return result([ids[-1]], False, "exact_gate", "All takes have the same words: keep the last.")
    raw = predict(state, build_questions(state))
    probs = {(k[len("keep_"):] if k.startswith("keep_") else k): float(v) for k, v in raw.items()}
    if set(probs) != set(ids):
        raise ValueError(f"model answered {sorted(probs)} for takes {ids}")
    d = _decide_probs(probs)
    keep = [u for u in ids if u in set(d["keep"])]
    if not keep and empty_fallback:
        return result([ids[-1]], d["unclear"], "empty_fallback", "The model kept no take: keeping the last take.", probs)
    if d["unclear"] and keep and unclear_default:
        return result([ids[-1]], True, "model", "The model is unsure: keeping the last take; please check this group.", probs, model_keep=keep)
    return result(keep, d["unclear"], "model", "The model keeps no take." if not keep else f"The model keeps {', '.join(keep)}.", probs)


# ================================================================================================================ download and the public API
MODE_PATTERNS = {
    "4bit": ["export_manifest.json", "head*", "decision_config.json", "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
             "chat_template.jinja", "adapter/*", "base_bnb4/*"],
    "fp16": ["export_manifest.json", "head*", "decision_config.json", "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
             "chat_template.jinja", "config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json"],
}
DEFAULT_REPO = "spywarex/retake-keeper-qwen3-4b"
_SHARED = ("head", "decision_config", "tokenizer", "vocab", "merges", "added_tokens", "special_tokens", "chat_template")


def verify_files(folder: Path, mode: str) -> list[str]:
    """SHA-256 of the files the mode uses against export_manifest.json. Returns the problems (empty = all good)."""
    manifest = json.loads((folder / "export_manifest.json").read_text(encoding="utf-8"))["files"]
    problems = []
    for rel, meta in manifest.items():
        if mode == "4bit" and not rel.startswith(("base_bnb4/", "adapter/") + _SHARED):
            continue
        if mode == "fp16" and rel.startswith(("base_bnb4/", "adapter/")):
            continue
        p = folder / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            problems.append(f"{rel}: SHA-256 mismatch")
    return problems


class RetakeKeeper:
    """Pick the take(s) to keep in a group of retakes.

        keeper = RetakeKeeper.from_pretrained()                        # downloads about 3.5 GB once (mode "4bit"), then runs locally
        r = keeper.decide([{"start": 12.0, "end": 14.1, "text": "So the main idea is"},
                           {"start": 15.0, "end": 18.0, "text": "So the main idea is that it works."}])
        r["keep"]       # ["2"]  (ids default to 1, 2, ... in the order given)
        keeper.unload()
    """

    def __init__(self, model: StandaloneDecisionModel, **policy):
        self.model, self.policy = model, policy

    @classmethod
    def from_pretrained(cls, repo_or_path: str = DEFAULT_REPO, mode: str = "4bit", device: str = "cuda", revision: Optional[str] = None, verify: bool = False,
                        gpu_memory_gb: Optional[float] = None, **policy) -> "RetakeKeeper":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        folder = Path(repo_or_path)
        if not folder.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise DecisionModelIncompatible("pip install huggingface_hub to download the model") from exc
            folder = Path(snapshot_download(repo_id=str(repo_or_path), revision=revision, allow_patterns=MODE_PATTERNS[mode]))
        if verify:
            problems = verify_files(folder, mode)
            if problems:
                raise DecisionModelIncompatible("model files do not match export_manifest.json: " + "; ".join(problems[:5]))
        return cls(StandaloneDecisionModel(folder, mode=mode, device=device, gpu_memory_gb=gpu_memory_gb), **policy)

    def score(self, takes: list[dict], before: Optional[dict] = None, after: Optional[dict] = None) -> dict[str, float]:
        """Raw p(yes = keep) per take id, no policy."""
        state = build_state(takes, before, after)
        raw = self.model.score(state, build_questions(state))
        return {k[len("keep_"):]: v for k, v in raw.items()}

    def decide(self, takes: list[dict], before: Optional[dict] = None, after: Optional[dict] = None) -> dict:
        """The hybrid policy (see decide_group). takes: dicts with start, end, text and an optional id."""
        return decide_group(takes, before, after, self.model.score, **self.policy)

    def unload(self) -> None:
        self.model.unload()

    def __enter__(self) -> "RetakeKeeper":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
