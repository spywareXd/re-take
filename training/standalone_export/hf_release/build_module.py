"""
Generate hf_release/retake_keeper.py: ONE self-contained file (transformers + torch + safetensors + huggingface_hub; bitsandbytes for 4-bit; no ReTake repo, no Unsloth)
= the standalone loader (decision_standalone.py, without the comparison-only 4bit_requant mode) + the training-time input builders + the hybrid keeper policy + a
from_pretrained() that downloads the model from the Hugging Face repo.

    python training/standalone_export/hf_release/build_module.py        # refuses to overwrite an existing retake_keeper.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from training.build_decision_data import decision_rules  # noqa: E402

OUT = HERE / "retake_keeper.py"
if OUT.exists():
    raise SystemExit(f"{OUT} exists; not overwriting (move it or pick another name)")
src = (REPO / "pipeline" / "step_06_llm_adjudication" / "decision_standalone.py").read_text(encoding="utf-8")
body = src[src.index("from __future__ import annotations"):]


def cut(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one occurrence of: {old[:70]!r}")
    return text.replace(old, new)


body = cut(body, 'REPO = Path(__file__).resolve().parents[2]\nDEFAULT_MODEL_DIR = REPO / "models" / "decision_standalone"\n', "")
body = cut(body, 'MODES = ("4bit", "fp16", "4bit_requant")', 'MODES = ("4bit", "fp16")')
body = cut(body, "self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR", "self.model_dir = Path(model_dir)")
body = cut(body, "model_dir: Optional[str | Path] = None, mode", "model_dir: str | Path, mode")
body = cut(body, 'if self.mode in ("4bit", "4bit_requant"):', 'if self.mode == "4bit":')
i = body.index("        else:                                                           # 4bit_requant")
j = body.index("        self._lm.config.use_cache = False")
body = body[:i] + body[j:]
body = cut(body, "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig", "from transformers import AutoModelForCausalLM, AutoTokenizer")
body = cut(body, "# ---------------------------------------------------------------------------------------------------------------------", "# ---------------------------------------------------------------------------------------------------------------------", ) if False else body

cues = json.loads((REPO / "pipeline" / "step_06_llm_adjudication" / "keeper_cues.json").read_text(encoding="utf-8"))
cue_list = list(cues["bank_cues"]) + [t for texts in cues.get("natural_keep_none_cues", {}).values() for t in texts]
tail = (HERE / "tail.py.in").read_text(encoding="utf-8").replace("__RULES__", repr(decision_rules())).replace("__CUES__", json.dumps(cue_list, ensure_ascii=False))

header = '''"""
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
'''
OUT.write_text(header + body.rstrip() + "\n" + tail, encoding="utf-8", newline="\n")
print("wrote", OUT, len(OUT.read_text(encoding="utf-8").splitlines()), "lines")
