# Standalone decision model: export spec

What the final keeper decision model is, read from the original (Unsloth/Studio) code and run folder. Nothing was modified to produce this.

## Model
- Backbone: Qwen3-4B-Instruct-2507 (`Qwen3ForCausalLM`, hidden 2560, tied embeddings). Trained on `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (dynamic NF4, double quant, block 64, bf16 compute; some projections stay bf16).
- LoRA: r16, alpha16 (scaling 1.0), 252 target weights (7 projections per layer), fp32 weights. Adapters are in `runs/keeper_loop/final_all/model/adapter_model.safetensors`.
- Head: Cloudflare Clef `JointSchemaHead` (hidden 2560, width 512, routing_layers 2, layers 4, heads 8, feedforward 2048; 32.05M params). Saved bf16 weights + fp32 scalars, loaded fp32. The fitted temperature (0.3923) is folded into the head, so the served temperature is 1.0.

## Scoring path (what Unsloth's `FastDecisionModel.predict` does)
1. `state`: a JSON-looking string is parsed to an object; questions are normalised (`noul` only here).
2. `encode_record` (Cloudflare reference, Apache-2.0): Qwen chat template, system prompt, `STATE:` as sorted compact JSON, `SCHEMA FIELDS` with one block per question, options ordered (true, false). Read length is max(4096, 16384) = 16384 tokens; a longer state is cut at its end.
3. Backbone `last_hidden_state` under bf16 autocast, head forward (lexical option vectors come from the tied lm_head rows), softmax over [true, false] -> `p_yes`.

## Files (`models/decision_standalone/`, gitignored except the manifest)
| file | use |
|---|---|
| `base_bnb4/`, `adapter/` | exact mode: the trained pre-quantized base + unmerged LoRA |
| `model-0000x-of-00002.safetensors`, `config.json` | fp16 merged weights: dequantize(NF4 base) + B@A in fp32, stored fp16 |
| `head.safetensors`, `head_config.json`, `decision_config.json` | copies of the original head files |
| `export_manifest.json` | versions, merge stats, SHA-256 of every file |

## Loader modes (`pipeline/step_06_llm_adjudication/decision_standalone.py`)
- `4bit` (recommended): base_bnb4 + adapters applied by forward hooks (no peft). Same weights as training.
- `fp16`: merged fp16. About 8 GB of weights, so on the 8 GB GPU part is offloaded to the CPU.
- `4bit_requant`: merged weights re-quantized to NF4. Comparison only; it throws away most of the small LoRA update (|dW|/|W| is about 0.3%).
- Attention is a small causal SDPA override (no T x T mask), needed to fit 16k tokens in 8 GB.
- Libraries: transformers >= 4.51 (tested 5.15.1), torch (tested 2.5.1+cu124), safetensors, bitsandbytes for the 4-bit modes.

## Environment changes
Installed into the repo venv (pure additions, torch untouched): `bitsandbytes==0.50.2`, `accelerate==1.15.0`. They are not recorded in pyproject/uv.lock.
