# Models

No weights are stored in git. Two trained models are used, both published on the Hugging Face Hub.

| Model | Where it is used | Size | Link |
|---|---|---|---|
| retake-pair-scorer | step 4, `models/final/final_scorer.py` | about 270 MB (3 cross-encoder seeds) | [spywarex/retake-pair-scorer](https://huggingface.co/spywarex/retake-pair-scorer) |
| retake-keeper-qwen3-4b | step 6, optional keeper | about 3.5 GB | [spywarex/retake-keeper-qwen3-4b](https://huggingface.co/spywarex/retake-keeper-qwen3-4b) |

## Pair scorer

`models/final/thresholds.json` (in git) holds the stacker coefficients, the thresholds and the unsure band. The cross-encoder seeds `ce_seed0`, `ce_seed1`, `ce_seed2` are fetched from the Hub into `models/final/` the first time `FinalScorer()` needs them (about 270 MB, once). Set `RETAKE_OFFLINE=1` to disable the download; the scorer then falls back to the plain stage-2 score with the old 0.65 / 0.35 rules and logs a warning. Use another copy of the weights with `RETAKE_PAIR_SCORER_REPO=<hub repo id>` or by passing `model_dir`.

Without this repository: the Hub repo contains `retake_pair_scorer.py`, a single file with the same scoring code.

## Keeper model

`models/decision_standalone/` is where a local copy of the keeper is expected (git-ignored apart from `export_manifest.json`). The keeper needs a CUDA GPU (about 5 GB of VRAM in 4-bit mode). The Hub repo contains `retake_keeper.py`, a single file that loads it without this repository or Unsloth. If you do not have a GPU, use `--keeper-provider keep-last` or a hosted provider.

Both models were trained on private recordings; see the data notice in [../data/synthetic/README.md](../data/synthetic/README.md) and the limits in the main README.

## Other files

`candidate.py`, `retake.py`, `segment.py`, `transcript.py`: the data models used by the pipeline. `final/`: the scorer, the scripts that fit it, and `thresholds.json`.
