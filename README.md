# re-take (retake-detection)

## Project Summary
This project aims to build an automated pipeline for detecting and removing retakes, restarts, stutters, and redundancies in spoken transcripts (from video or audio sources). 

When creators record content, they often stumble, correct themselves, or restart a sentence. This tool analyzes transcription data (including metadata like timestamps, speaker IDs, and transcription confidence) to identify these overlapping or corrective segments. Ultimately, this enables an automated editing process that cuts out the "bad" takes and keeps only the final, clean versions.

## Setup (uv)
Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`); there is no `requirements.txt`. FFmpeg must be on PATH.

```
uv sync                              # create .venv and install the locked dependencies
uv run pytest tests models/final     # run the tests
uv run python -m service             # local HTTP service for the editor (see docs/frontend_integration/)
uv add <package>                     # add a dependency (updates pyproject.toml and uv.lock)
```

## Core Objectives
- **Parse raw transcripts**: Ingest JSON transcript data with word/segment level metadata, timestamps, and confidence scores.
- **Detect redundancies**: Use a combination of token matching and semantic similarity to find repeated or corrected phrases within a local window.
- **Identify retake boundaries**: Pinpoint exactly where a mistake was made and where the clean take begins.
- **Automated Decision Making**: Leverage LLMs and advanced heuristics to adjudicate ambiguous overlaps, score confidence, and finalize the "cut" decisions.
- **Video/Audio Editing**: Generate automated edit lists to programmatically trim the original media files (e.g., via FFmpeg).

## Pipeline Architecture
The system is designed around a modular, 8-stage pipeline:
1. **Transcription (`step_01_transcription`)**: Extracting audio and generating base transcripts with timestamps.
2. **Preprocessing (`step_02_preprocessing`)**: Cleaning, tokenizing, and chunking the raw text.
3. **Token Matching (`step_03_token_matching`)**: Finding exact or near-exact word repetitions using n-grams and edit distance.
4. **Scoring (`step_04_semantic_matching`)**: Every candidate pair is scored by the *final scorer*: a logistic-regression **stacker** over the plain Stage 2 score (`0.50*semantic + 0.30*lexical + 0.20*containment`, built from the pair's own fields) and the logit of a fine-tuned **cross-encoder** (3-seed average). Pairs with stacker probability `>= high` are retakes, `< low` are not, and the pairs in between are resolved by the cross-encoder (`>= t_amb` is a retake). Every pair ends as `retake` or `not_retake`. The old recency-adjusted score and class are kept only as `legacy_stage2_score` / `legacy_class`. See [docs/scorer.md](docs/scorer.md).
5. **Span Aggregation (`step_05_span_aggregation`)**: Clustering ONLY the pairs decided `retake` (union-find) into spans. Every span is already confirmed (`needs_adjudication = False`) and needs a keeper (`needs_keeper = True`); each records its unit count and weakest stacker probability, and spans of more than 4 units are logged.
6. **Keeper selection (`step_06_llm_adjudication`)**: Every span goes to the LLM, which is told the span is already a confirmed retake group and only chooses the unit to keep and the units to remove. A span the LLM cannot resolve (no API key, API error, invalid answer) is marked `failed` and is **not cut**.
7. **Review (`step_07_review`)**: Exporting reports and structured JSON for human review or debugging.
8. **Video Edit (`step_08_video_edit`)**: Translating the finalized cuts into FFmpeg commands to produce the final, polished media.
