# re-take

re-take finds the retakes in a spoken recording: the places where a speaker stumbled, corrected themselves or said the same line again. It finds and groups those repeated sentences. It never decides what to cut by itself, and it never calls a language model to edit anything: the person (or an optional keeper model) picks the take to keep, and everything else is cut.

This repository has two parts:

- **The retake pipeline** (Python, `pipeline/`, `models/`, `service/`): transcription, sentence splitting, retake detection, grouping, an optional "which take to keep" suggestion, and a local HTTP service.
- **Retake Editor** (Rust, `retake-editor/`): a small video editor that uses the service. You review the groups, tick the sentences to keep, and everything not kept is cut in one undoable step.

## How it works

```
video/audio
  1 transcription     WhisperX / faster-whisper, word timestamps           pipeline/step_01_transcription
  2 preprocessing     split into sentence units (pauses, punctuation)      pipeline/step_02_preprocessing
  3 token matching    exact / near-exact / n-gram / edit-distance          pipeline/step_03_token_matching
  4 pair scoring      stage-2 score + fine-tuned cross-encoder + stacker   models/final, pipeline/step_04_semantic_matching
  5 grouping          union-find over pairs decided "retake"               pipeline/step_05_span_aggregation
  6 keeper (optional) suggests which take to keep                          pipeline/step_06_llm_adjudication
  7 review            JSON documents for the editor                        pipeline/step_07_review
  8 video edit        ffmpeg cuts (cut modes: anchored, retake_only, sentences)   pipeline/step_08_video_edit
```

Pairs the scorer is unsure about ("ambiguous") are asked to the user in the editor, and the user's answers are final for those pairs. Scores are not probabilities: show them as a lean ("retake" / "different"), not a percentage. Details: [docs/pipeline.md](docs/pipeline.md).

## Models

Weights are not stored in git. They are downloaded from the Hugging Face Hub the first time they are needed.

| Model | Used for | Link |
|---|---|---|
| retake-pair-scorer (3 cross-encoder seeds + stacker thresholds) | step 4: retake / not retake / ambiguous | [spywarex/retake-pair-scorer](https://huggingface.co/spywarex/retake-pair-scorer) |
| retake-keeper-qwen3-4b (Qwen3-4B QLoRA + decision head) | step 6: optional keeper suggestion | [spywarex/retake-keeper-qwen3-4b](https://huggingface.co/spywarex/retake-keeper-qwen3-4b) |

Both are usable without this repository through the single-file loaders in their model cards. Offline use: download the files once, or set `RETAKE_OFFLINE=1` to stop the scorer from trying to download (it then falls back to the plain stage-2 score and logs a warning).

## Install

Requirements: Python 3.11 to 3.13, [uv](https://docs.astral.sh/uv/), FFmpeg on `PATH`. A CUDA GPU is only needed for transcription with a large model and for the keeper model; scoring runs on CPU.

```
uv sync
uv run pytest tests models/final
```

`uv.lock` is resolved for Windows x64 with CUDA 12.4 builds of PyTorch (see `pyproject.toml`). On Linux or macOS, install the same packages into your own environment (the dependency list is in `pyproject.toml`, plus a PyTorch build for your platform); this has not been tested here.

## Try it in a minute (no video, no GPU)

```
uv run python examples/quickstart.py          # downloads the pair scorer once (about 270 MB)
uv run python examples/quickstart.py --mock   # fully offline, stand-in scorer
```

It reads `examples/sample_transcript.json` (a small synthetic transcript with two retakes), splits it into sentences, finds candidate pairs and prints which pairs are retakes.

## Run on a video

```
uv run python -m pipeline.run_pipeline path/to/video.mp4 --cut-mode anchored
```

Useful flags: `--provider whisperx|whisper|assemblyai`, `--model large-v2`, `--device cuda|cpu`, `--keeper-provider finetuned|gpt-oss|keep-last`, `--cut-mode sentences|anchored|retake_only`. The keeper step needs either the local keeper model or an API key in `.env` (`GEMINI_API_KEY`, `GROQ_API_KEY`); with `--keeper-provider keep-last` nothing external is used. See [service/README.md](service/README.md) and [docs/api/cut_modes.md](docs/api/cut_modes.md).

To get only a subtitle file of the sentences that would be kept (nothing is cut, the timeline is not shortened): `uv run python -m pipeline.export_keeper_srt path/to/video.mp4`.

## Service and editor

`uv run python -m service` starts the local service on `127.0.0.1:8765`. The editor talks to it over HTTP: [service/README.md](service/README.md), contract in [docs/api/](docs/api/). The editor is built with Cargo: [retake-editor/README.md](retake-editor/README.md).

## What the data and results do and do not show

- **Data.** The scorer and the keeper were trained and evaluated on labelled sentences from private recordings that we do not have the right to publish. **Only the synthetic part of the data is published, so the published dataset is incomplete** and the training cannot be reproduced from this repository. See [data/synthetic/README.md](data/synthetic/README.md).
- **Pair scorer.** On four held-out folds, with thresholds chosen on the other folds, the retake tier had precision 0.90 and recall 0.84, and 10.5% of pairs were left ambiguous. The data covers few speakers; the weakest fold had precision 0.75.
- **Keeper model.** On 131 labelled groups (one model per outer fold; folds 3 and 4 were also the set the recipe was tuned on) the fine-tuned model alone gets the exact keeper set right in 114 groups, the keep-last baseline in 117 and the hosted Groq model in 108. Against Groq and against keep-last the paired bootstrap intervals touch or cross zero (see `docs/figures/f3_paired_differences.png`), so there is no reliable difference from either: the model is on par with Groq and does not beat keep-last. The default "hybrid" (model plus rules such as the keep-last fallback) scores 118 of 131, but its rules were chosen after seeing these results, so treat that number as optimistic. A final model trained on all 131 groups has no held-out estimate; its accuracy on its own training groups is not a performance figure. Charts: [docs/figures/](docs/figures/).
- **Cut modes** are editing choices, not accuracy claims.

## Tests

`uv run pytest tests models/final` runs offline. Tests that need private data, a video, a GPU or the downloaded weights are skipped with a reason. With `RETAKE_OFFLINE=0 uv run pytest models/final` the scorer downloads the real weights and the smoke test that loads them runs; the other scorer tests need artifacts of the private training run and stay skipped.

## Repository layout

| Path | What |
|---|---|
| `pipeline/` | the eight pipeline steps |
| `models/` | data models, the final pair scorer, thresholds |
| `service/` | local FastAPI service for the editor |
| `retake-editor/` | the Rust editor |
| `training/`, `data/training/` | training and evaluation scripts (they need the private data) |
| `data/synthetic/` | the published synthetic data and its generators |
| `docs/` | pipeline notes, API contract, figures, screenshots |
| `examples/` | the quickstart and its synthetic sample |

## License and citation

MIT, see [LICENSE](LICENSE). To cite, see [CITATION.cff](CITATION.cff).

Built on faster-whisper / WhisperX, sentence-transformers (`all-MiniLM-L6-v2`), `cross-encoder/ms-marco-MiniLM-L-6-v2`, Qwen3-4B-Instruct-2507, Cloudflare's Clef decision head (Apache-2.0) and the FilmCraft media crates (MIT OR Apache-2.0).
