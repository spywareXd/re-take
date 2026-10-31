# Pipeline

The pipeline finds and groups sentences that are repeats of each other. It never says what to keep and never asks a language model to edit. Each step reads the previous step's output; `pipeline/run_pipeline.py` runs them in order and the local service (`service/`) runs the same code for the editor.

| Step | Folder | In | Out |
|---|---|---|---|
| 1 Transcription | `step_01_transcription` | video or audio | transcript with word timestamps (`models/transcript.py`) |
| 2 Preprocessing | `step_02_preprocessing` | transcript | sentence units (`ComparableUnit`) |
| 3 Token matching | `step_03_token_matching` | units | candidate pairs with lexical scores |
| 4 Pair scoring | `models/final`, `step_04_semantic_matching` | candidate pairs | pair decisions: retake, not retake, ambiguous |
| 5 Span aggregation | `step_05_span_aggregation` | decided pairs | groups of units that repeat one another |
| 6 Keeper (optional) | `step_06_llm_adjudication` | groups | a suggested keeper per group |
| 7 Review | `step_07_review` | everything above | JSON documents for the editor, user answers |
| 8 Video edit | `step_08_video_edit` | keep decisions | cut list and the edited video (ffmpeg) |

## Step by step

**1 Transcription.** WhisperX or faster-whisper (or AssemblyAI) with word timestamps. The model size is chosen from the models installed locally; the compute type falls back automatically (for example from float16 to int8) when the GPU is small.

**2 Preprocessing.** Sentences are cut inside each ASR segment by pauses longer than the silence gap (default 0.7 s, range 0.10 to 3.00 s), by sentence-final punctuation and by speaker change. Segments are never merged with each other. Text is normalised (lower case, contractions expanded, punctuation removed) for matching, and the raw text is kept for display. Changing the gap re-splits from the saved word timestamps without transcribing again.

**3 Token matching.** Every unit is compared with the units that follow it inside a time window (default 300 s, recency-weighted) using sequence ratio, Jaccard, containment, bigram overlap, exact and near-exact match (fillers removed) and a restart-phrase check. Pairs above a low lexical threshold (0.10) become candidates. An optional detector flags restarts inside one unit, as a suggestion only.

**4 Pair scoring.** The plain stage-2 score is `0.50 * semantic + 0.30 * lexical + 0.20 * containment`, with the semantic part the cosine of `all-MiniLM-L6-v2` embeddings mapped to [0, 1]. A fine-tuned cross-encoder (three seeds, probabilities averaged) reads the two raw sentences. A logistic-regression stacker combines the stage-2 score with the cross-encoder logit. `p >= 0.866` is a retake, `p < 0.171` is not, and in between the cross-encoder decides: probability `>= 0.969` retake, `<= 0.32` not a retake, otherwise ambiguous. Thresholds and coefficients are in `models/final/thresholds.json`; see [scorer.md](scorer.md). Scores are not calibrated probabilities.

**5 Span aggregation.** Only pairs decided "retake" are clustered (union-find, with the user's answers as fixed constraints). Each group is a set of takes of one line.

**6 Keeper.** Optional, and only run when the user asks. The default keeper is the fine-tuned model plus rules: all takes being short cue phrases ("here we go") keeps none, identical takes keep the last, and if the model keeps nothing or is unsure the last take is kept and the group is flagged. Other options are a hosted model (`gpt-oss` through Groq) or plain keep-last. See [keeper/pipeline_update.md](keeper/pipeline_update.md) and [keeper/hybrid_comparison.md](keeper/hybrid_comparison.md).

**7 Review.** The service produces candidate and group documents for the editor ([api/](api/)). The user answers the ambiguous pairs, then ticks the sentences to keep; the answers are final for those pairs.

**8 Video edit.** Everything not kept is cut. Cut modes: `anchored` (extend each removal into the adjacent dead air, the default), `retake_only` (cut only the removed words) and `sentences` (keep only spoken sentences and remove silences); see [api/cut_modes.md](api/cut_modes.md). Cuts are rendered in one ffmpeg pass with a 15 ms audio fade at every join.

## Data on disk

With a project folder `X/`, the editor stores `X/X.<ext>` and `X/data/` (transcript, units, candidates, groups, keeper advice, progress log, exports). Results are written atomically; replaced results are kept as numbered copies in `X/data/stale/`. Details: [app_changes.md](app_changes.md).
