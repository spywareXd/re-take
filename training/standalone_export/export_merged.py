"""
Export the final decision model to a NEW folder, additive only (run with the STUDIO venv: torch + bitsandbytes + safetensors):

    <studio python> -X utf8 training/standalone_export/export_merged.py [--out models/decision_standalone] [--dtype float16]

The final model is a QLoRA model: LoRA adapters trained on Unsloth's pre-quantized checkpoint (unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit: dynamic NF4,
some projections kept in bf16) plus a joint decision head. This script merges the adapters ONCE into the weights the model was trained on, in float32:
    W_merged = dequantize(NF4 base weight, from its stored quant state)  (or the bf16 weight where the checkpoint keeps it unquantized)  +  scaling * (B @ A)
and stores W_merged as float16 (more mantissa than bf16: the LoRA update is small relative to W). Every other tensor is the checkpoint's tensor, cast to the same dtype.
No file under runs/, the Studio folders or the Hugging Face cache is modified; the out folder must not exist yet (nothing is overwritten).

Writes into --out:
    model-0000x-of-0000N.safetensors, model.safetensors.index.json, config.json, generation_config.json, tokenizer files   (merged HF model, Qwen3ForCausalLM)
    head.safetensors, head_config.json, decision_config.json                                                                (copies of the original head files)
    adapter/ (adapter_model.safetensors, adapter_config.json) and base_bnb4/ (the pre-quantized checkpoint)                  (copies: the exact 4-bit mode, no merge, no re-quantization)
    export_manifest.json                                                                                                    (source, versions, sizes, SHA-256 of every output)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs" / "keeper_loop" / "final_all" / "model"
HF = Path.home() / ".cache" / "huggingface" / "hub"
SHARD_BYTES = 4_500_000_000


def snapshot(repo_id: str) -> Path:
    base = HF / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(p for p in base.iterdir() if p.is_dir())
    if not snaps:
        raise SystemExit(f"no cached snapshot for {repo_id}")
    return snaps[-1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "models" / "decision_standalone"))
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty: refusing to overwrite (choose a new --out)")
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    import transformers

    preq, orig = snapshot("unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"), snapshot("Qwen/Qwen3-4B-Instruct-2507")
    acfg = json.loads((SRC / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"]) if not acfg.get("use_rslora") else float(acfg["lora_alpha"]) / float(acfg["r"]) ** 0.5
    dev = torch.device("cuda")

    # LoRA pairs: base_model.model.<module>.lora_A.weight / lora_B.weight  ->  <module>.weight
    lora: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(SRC / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
            if not m:
                raise SystemExit(f"unexpected adapter tensor {k}")
            lora.setdefault(m.group(1) + ".weight", {})[m.group(2)] = f.get_tensor(k)
    print(f"adapter: {len(lora)} target weights, r={acfg['r']}, alpha={acfg['lora_alpha']}, scaling={scaling}")

    weights: dict[str, torch.Tensor] = {}
    shards, current, size = [], {}, 0
    n_merged = n_dequant = n_plain = 0
    stats = []
    t0 = time.time()
    with safe_open(str(preq / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        companions = {k for k in keys for other in keys if other != k and k.startswith(other + ".")}
        for name in sorted(k for k in keys if k not in companions):
            w = f.get_tensor(name)
            if w.dtype == torch.uint8:                                   # NF4 weight: dequantize with its stored quant state
                qs = {k[len(name) + 1:]: f.get_tensor(k) for k in keys if k.startswith(name + ".")}
                state = F.QuantState.from_dict(qs, device=dev)
                full = F.dequantize_4bit(w.to(dev), state).float()
                n_dequant += 1
            else:
                full = w.to(dev).float()
                n_plain += 1
            if name in lora:
                a, b = lora[name]["A"].to(dev).float(), lora[name]["B"].to(dev).float()
                delta = (b @ a) * scaling
                if delta.shape != full.shape:
                    raise SystemExit(f"{name}: LoRA delta {tuple(delta.shape)} does not match the weight {tuple(full.shape)}")
                stats.append((name, float(delta.norm() / full.norm())))
                full = full + delta
                n_merged += 1
            current[name] = full.to(dtype).cpu().contiguous()
            size += current[name].numel() * current[name].element_size()
            if size >= SHARD_BYTES:
                shards.append(current)
                current, size = {}, 0
    if current:
        shards.append(current)
    missing = sorted(set(lora) - {k for s in shards for k in s})
    if missing:
        raise SystemExit(f"adapter targets without a base tensor: {missing[:3]}")
    print(f"merged {n_merged} weights; {n_dequant} NF4 tensors dequantized, {n_plain} plain tensors; {len(shards)} shard(s); {time.time() - t0:.0f}s")
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for s in shards for t in s.values())}, "weight_map": {}}
    for i, s in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, str(out / fname), metadata={"format": "pt"})
        index["weight_map"].update({k: fname for k in s})
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    cfg = json.loads((orig / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = args.dtype
    cfg["dtype"] = args.dtype
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"):
        for d in (SRC, orig, preq):
            if (d / name).is_file():
                shutil.copyfile(d / name, out / name)
                break
    # the original head and its config (copied, not moved)
    shutil.copyfile(SRC / "joint_head.safetensors", out / "head.safetensors")
    shutil.copyfile(SRC / "joint_head_config.json", out / "head_config.json")
    shutil.copyfile(SRC / "unsloth_decision_config.json", out / "decision_config.json")
    (out / "adapter").mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(SRC / name, out / "adapter" / name)
    (out / "base_bnb4").mkdir()
    for p in preq.iterdir():
        if p.is_file() and p.name not in (".gitattributes", "README.md"):
            shutil.copyfile(p, out / "base_bnb4" / p.name)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "source_run": "runs/keeper_loop/final_all (keeper_loop/train_decision.py, 2026-10-27); model folder = adapters + head",
        "base_model_id_in_adapter": acfg["base_model_name_or_path"], "base_trained_on": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit (dynamic NF4, double quant, bf16 compute)",
        "base_snapshot_preq": preq.name, "base_snapshot_original": orig.name,
        "merge": {"formula": "dequantize(base) + scaling * (B @ A) in float32, stored as " + args.dtype, "scaling": scaling, "weights_merged": n_merged,
                  "nf4_tensors_dequantized": n_dequant, "plain_tensors": n_plain, "relative_update_norm_|dW|/|W| (median, max)": [sorted(v for _, v in stats)[len(stats) // 2], max(v for _, v in stats)]},
        "export_dtype": args.dtype, "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "bitsandbytes": bnb.__version__},
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {str(p.relative_to(out)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
        "notes": ["decision_config.json carries folded_temperature 0.3923 (already folded into the head's logit scales; served temperature is 1.0)",
                  "head.safetensors is the original (bf16 weights, float32 scalars); the loader casts it to float32"],
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", out, "| total GB:", round(sum(p.stat().st_size for p in files) / 1e9, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
