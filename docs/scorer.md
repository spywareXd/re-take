# The final scorer (Stages 4 to 6)

This page describes how the pipeline decides which pairs of spoken units are retakes, how ambiguous pairs are resolved, and how the LLM is used. The scorer lives in `models/final/` and is called from `pipeline/step_04_semantic_matching/final_stage4.py`.

## 1. What happens to a pair

```
step 03 candidate pair
   |  run_stage2 (unchanged) -> semantic_score, and the OLD recency-adjusted score / class
   v
plain_stage2_score = 0.50*semantic_score + 0.30*lexical_score + 0.20*containment     (the pair's own fields)
ce_logit           = logit(clip(mean over 3 seeds of sigmoid(cross-encoder logit)))    (raw text, earlier unit first, max length 128, batch 64)
stacker_p          = sigmoid(intercept + sum_i coef_i * (x_i - mean_i) / scale_i)      x = [plain_stage2_score, ce_logit]
   |
   |  stacker_p >= high  -> retake          (resolved_by = "stacker")
   |  stacker_p <  low   -> not_retake      (resolved_by = "stacker")
   |  otherwise (ambiguous): cross-encoder probability >= t_amb -> retake, else not_retake   (resolved_by = "cross_encoder")
   v
every pair is "retake" or "not_retake"
```

The plain Stage 2 score is **not** the recency-adjusted score the old pipeline reported: the stacker and the thresholds were trained on the plain formula. The old values are stored as `legacy_stage2_score` and `legacy_class` for audit, and the old key names `stage2_score` / `classification` are removed from candidates so nothing can read them by accident. The old lexical-vs-semantic disagreement rule is no longer used for classification; its value stays stored (`lexical_semantic_disagreement`, `high_disagreement_flag`).

## 2. Saved coefficients and thresholds (`models/final/thresholds.json`)

| item | value |
|---|---|
| features | `plain_stage2_score`, `ce_logit` (logit clipped at 1e-6) |
| scaler mean / scale | Stage 2: 0.46112 / 0.13722; CE logit: -7.44436 / 6.39359 |
| standardized coefficients | +1.80829 (Stage 2), +1.38016 (cross-encoder) |
| intercept, C | -3.16164, C = 0.1 (L2, class_weight balanced) |
| `high` | 0.86567 (lowest threshold with precision >= 0.95 on the out-of-fold stacker probabilities) |
| `low` | 0.17134 (highest threshold with recall >= 0.98 on the same probabilities) |
| `t_amb` | 0.96933, fitted once from the saved out-of-fold scores on the 376 pairs whose out-of-fold stacker probability lies in [low, high) (36 retakes): the cross-encoder threshold that maximises F1 |
| cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on `pairs_v2.csv` (3 seeds: `models/final/ce_seed0..2`) |
| embedding model | `all-MiniLM-L6-v2` (semantic score) |
| versions | scorer `stacker_cross_encoder`, model `final-ce-3seed-v2labels+stacker-v1`, feature `v1`, labels `pairs_v2 (sha256 d34ece717418)` |

Every candidate and the JSON envelope of `semantic_candidates.json` carry `scorer_name`, `thresholds_file`, `model_version`, `feature_version`, `label_version`; each candidate also carries `plain_stage2_score`, `ce_logit`, `stacker_p`, `class`, `resolved_by`, `legacy_stage2_score` and `legacy_class`.

## 3. From pairs to the cut list

- **Stage 5** unions ONLY pairs whose final `class` is `retake` (union-find, sibling short-unit attachment and the likely-final-unit heuristic are unchanged). A pair decided `not_retake` never bridges two spans. Every span has `needs_adjudication = False`, `needs_keeper = True`, `n_units`, `min_stacker_p` (its weakest edge) and `decision_source` (`stacker`, `cross_encoder` or `mixed`). A warning is logged for spans with more than 4 units: long chains may merge separate retakes.
- **Stage 6** sends **every** span to the LLM, but only to choose the keeper. The prompt states that the span is already a confirmed repeated-attempts group and tells the model not to judge whether it is a retake; it returns `keeper_unit_id`, `units_to_remove`, `confidence` and `reasoning`. `is_genuine_retake_group` is set to true by the code. The existing client/provider abstraction, retries (2), JSON validation (keeper belongs to the span, removals belong to the span, keeper not in the removal list) and failure handling are reused; provider and model come from `ADJUDICATION_PROVIDER` / `ADJUDICATION_MODEL` (default `gemini` / `gemini-2.5-flash`) and the key from the environment.
- A span the LLM cannot resolve (no API key, API failure, invalid answer after the retries) gets `adjudication_status = "failed"`, is recorded in `adjudicated_spans.json`, and is **not cut**; a warning says so. Nothing raises.
- **Cut list** (`step_08_video_edit/cut_generator.py`) cuts only spans with status `resolved` and a non-empty removal list.

## 4. Fallbacks (robustness only, not a mode)

If `thresholds.json` or the weights are missing or fail to load, the scorer logs a warning and uses plain Stage 2 with the tuned thresholds already saved in `data/training/results_v2/final/fair_comparison.json` (currently low 0.5095, high 0.6757); `scorer_name` becomes `tuned_stage2_fallback`. If that file is missing too it uses the legacy 0.65 / 0.35 rules (`legacy_stage2_fallback`). In both fallbacks a pair between the thresholds is **not cut** (`not_retake`, `resolved_by = "stage2_fallback"`): there is no cross-encoder to resolve it.

## 5. Configuration

`run_full_pipeline(..., scorer_model_dir=None, scorer_device=None, scorer_batch_size=64)`: the folder holding `thresholds.json` and `ce_seed*/` (default `models/final`), the device for the cross-encoder (GPU if present, else CPU) and its batch size. `FinalScorer(model_dir, device, batch_size, ...)` takes the same options.

### 5b. Stage 3 candidate settings (aligned with the export)

The pipeline's Stage 3 defaults (`TokenMatchingConfig` in `pipeline/step_03_token_matching/config.py`) are now `candidate_threshold = 0.10` (minimum lexical score) and `window_seconds = 300.0`, the same values `training/populate_pairs.py` used to export `pairs_v1` / `pairs_v2` (`training/config.py`: `DEFAULT_MIN_LEXICAL`, `DEFAULT_WINDOW_SECONDS`). The old defaults were 0.35 and 120 s and are noted in comments. Nothing else differs: the exporter overrides only these two settings, and minimum tokens (3), recency decay (45 s), the lexical weights, the 0.7 s pause split and the restart/filler lexicons are already the defaults. Both remain configurable per run.

Effect on cached `speaker_a_3` (candidate generation plus Stage 4 on a warm GPU with the scorer already loaded; mock LLM, no API calls):

| | old defaults (0.35 / 120 s) | new defaults (0.10 / 300 s) |
|---|---|---|
| Stage 3 candidates | 21 | 86 |
| in `pairs_v2.csv` (86 rows, 69 labelled 0/1, 22 retakes) | 21 of 86; 21 of 69 labelled; 20 of 22 retakes | 86 of 86; 69 of 69 labelled; 22 of 22 retakes |
| Stage 4 runtime | 0.08 s (3.8 ms/pair) | 0.24 s (2.8 ms/pair) |
| spans / largest / cut | 4 / 6 units / 12 discards, 23.3 s | 4 / 6 units / 12 discards, 23.3 s |

The extra candidates are all low-overlap or distant pairs, so they cost little (about 0.2 s here); the plain Stage 2 score of every one of the 86 pairs matches `pairs_v2.csv` within 0.0004. Candidate generation itself takes about 0.01 s in both cases. These numbers are an engineering check, not an evaluation.

## 6. Retraining

1. Put the corrected labels in `data/training/pairs_v2.csv` and rerun the evaluation scripts (`baselines_v2.py`, `train_cross_encoder_v2.py`, ...) so that `results_v2/baseline_scores.csv` and `ce_scores.csv` hold fresh out-of-fold scores.
2. `python models/final/fit_final.py` (stacker, high/low thresholds, `final_oof_scores.csv`; it rewrites `thresholds.json`).
3. `python models/final/train_final_ce.py` (final cross-encoder, 3 seeds, writes `ce_seed0..2/`; needs a GPU for sensible speed).
4. `python models/final/fit_t_amb.py --refit` (adds `t_amb` and the version fields to `thresholds.json`; step 2 removes them, so always run this last).
5. `pytest tests models/final`.

## 7. Weights are not in git

The three cross-encoder folders (`models/final/ce_seed0..2/`, about 264 MB in total) are ignored by git. Download location: **TODO: add the release / Hugging Face URL once published** (publish with a model card and check the base model's license first). `thresholds.json` and all code are tracked. Without the weights the scorer falls back as in section 4 and logs a warning.

## 8. Measured performance (state once, with caveats)

At the high threshold the retake tier had out-of-sample precision of about 0.90 and recall of about 0.84 (nested over 4 speaker-grouped folds; 15 videos from 8 speakers, 340 retakes). The gains over a Stage 2 whose thresholds were tuned the same way are **within the noise**: recall +4.4 points (95% CI [-2.0, +9.5]), precision -4.5 points, and at matched precision targets the stacker's recall advantage did not have a confidence interval above zero. Per-fold results vary widely (precision 0.75 in the weakest fold), the cross-encoder out-of-fold scores are mildly optimistic, and performance on new speakers is unknown. Treat the numbers as indicative.

## 9. Known limits

- The Stage 3 defaults now match the export (see section 5b). If someone runs the pipeline with stricter settings than 0.10 / 300 s, low-overlap paraphrased retakes may never reach Stage 4: on `speaker_a_3` the old defaults (0.35 / 120 s) let only 21 of the 86 labelled candidate pairs through and missed 2 of its 22 retakes. The scorer is not the limiting factor there.
- The dataset column `passes_stage1_default` (written by `training/populate_pairs.py` from the pipeline's default config) was computed with the OLD defaults (0.35 / 120 s): 394 of the 3,627 real rows carry 1. Because the defaults changed, a future refresh or export would set it to 1 for every pair. The exporter was not changed; pin the old values in `populate_pairs.py` before refreshing if the column's old meaning matters.
- The fallback modes cannot resolve ambiguous pairs, so they cut fewer pairs than the full scorer.
- `step_05_cutlist/` (an older, unused builder) still reads the old `classification` field and has not been adapted.
