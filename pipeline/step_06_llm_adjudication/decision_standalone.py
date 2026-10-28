"""
Standalone loader for the final keeper decision model: transformers + torch + safetensors (+ bitsandbytes for the 4-bit modes). No Unsloth, no Studio.

The model is a Qwen3-4B-Instruct-2507 backbone (QLoRA-fine-tuned) with Cloudflare's Clef joint-schema decision head. How it is defined is written down in
docs/finetune_final/standalone_export_spec.md; this file follows that spec. The record encoding and the head are re-implemented from Cloudflare's
`joint_schema_model.py` (Apache-2.0, https://huggingface.co/Cloudflare/clef), which Unsloth vendors unmodified.

    from pipeline.step_06_llm_adjudication.decision_standalone import StandaloneDecisionModel
    with StandaloneDecisionModel(mode="4bit") as m:                                   # loads lazily on the first score()
        p = m.score(state, questions)                                                 # {"keep_<unit id>": p_yes}

Modes (the files live in models/decision_standalone/, written by training/standalone_export/export_merged.py):
    "4bit"          the pre-quantized base the model was trained on (base_bnb4/) + the LoRA adapters applied in 16 bits on top (adapter/). Same weights as training and as
                    Unsloth's serving; no merge, no re-quantization. Needs bitsandbytes and a CUDA GPU (about 4.5 GB of VRAM).
    "fp16"          the LoRA merged into the dequantized base and stored as float16 (the model's root files). About 8 GB of weights: on an 8 GB GPU part of it is offloaded
                    to the CPU (accelerate), so it is slow there; on a larger GPU it runs fully on the GPU.
    "4bit_requant"  the fp16 merged weights quantized to NF4 on load. Provided for comparison only: re-quantizing the merged weights discards most of the small LoRA update.
Compute is bf16 autocast for the backbone and the head, as in Unsloth's serving path; the head weights are kept in float32.
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO / "models" / "decision_standalone"
MODES = ("4bit", "fp16", "4bit_requant")
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

    def __init__(self, model_dir: Optional[str | Path] = None, mode: str = "4bit", device: str = "cuda", max_length: int = SERVE_MAX_LENGTH,
                 gpu_memory_gb: Optional[float] = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
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
        if self.mode in ("4bit", "4bit_requant"):
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        else:                                                           # 4bit_requant
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self._lm = AutoModelForCausalLM.from_pretrained(str(d), quantization_config=quant, device_map={"": dev.index if dev.index is not None else 0}, dtype=torch.bfloat16, attn_implementation=attn)
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
