# Pipeline update: the fine-tuned keeper model as the default Step 6 provider

Status 2026-10-28: implemented and tested on branch `finetune/qwen-final`, not committed.

## 1. What changed and why

Step 6 (keeper selection) used to call a hosted LLM; the service used `openai/gpt-oss-120b` on Groq when a key was present. It now has two **keeper providers**:

| provider | what it is | needs |
|---|---|---|
| `finetuned` (**default**) | the fine-tuned Qwen3-4B decision model behind a hybrid rule policy (section 3) | Unsloth Studio running on this machine with the model loaded; no API key, no network beyond localhost |
| `gpt-oss` | the existing keeper call, unchanged (same prompt, same settings, same fail-closed handling) | an LLM key (`GROQ_API_KEY` for gpt-oss) |

The reason is to remove the per-group API call and its rate limits, failures and cost. It is **not** an accuracy gain: on the 131 labelled natural groups the model alone ties Groq and is slightly below keep-last (section 5).

## 2. Selecting a provider

Precedence: request field or CLI flag, then the environment variable, then the default (`finetuned`).

| where | how |
|---|---|
| config default | `KeeperConfig.provider` in `pipeline/step_06_llm_adjudication/keeper_policy.py` (`DEFAULT_PROVIDER = "finetuned"`) |
| environment | `RETAKE_KEEPER_PROVIDER=finetuned` or `gpt-oss` (pipeline and service) |
| CLI | `python -m pipeline.run_pipeline video.mp4 --keeper-provider gpt-oss` (`--llm-provider` / `--llm-model` default to `groq` / `openai/gpt-oss-120b` on the CLI) |
| Python | `run_full_pipeline(..., keeper_provider="gpt-oss")` (with `llm_provider` / `llm_model` as before) |
| service request | `POST /v1/adjudicate {..., "keeper_provider": "finetuned" | "gpt-oss"}` |

A provider chosen explicitly (request, CLI or argument) that cannot run returns a clear error (service: HTTP 400 naming the provider and the reason; pipeline: `ProviderUnavailable`). Providers are never swapped silently. When the **default** provider is unavailable in a pipeline run, the run continues, logs an ERROR, and every group falls back to keep-last (the model-error rule).

`GET /v1/capabilities` now also returns, for example:

```json
{"adjudicate": true, "refine": true,
 "keeper": {"default": "finetuned",
            "providers": {"finetuned": {"available": true, "reason": "decision model 'clef-ft:Qwen_final' via http://127.0.0.1:8888/v1/systemone"},
                          "gpt-oss": {"available": true, "reason": "LLM provider 'groq'"}}}}
```

`adjudicate` is true when the **default** provider is available.

**Roll back to gpt-oss by default:** set `RETAKE_KEEPER_PROVIDER=gpt-oss` (one line in `.env` or the shell), or change `DEFAULT_PROVIDER` to `"gpt-oss"` in `keeper_policy.py`.

## 3. The finetuned provider's policy

Applied in this order; each layer has its own flag (all on by default) in `KeeperConfig` and an environment override.

| layer | rule | flag / env | `decided_by` |
|---|---|---|---|
| a. cue rule (strong cue group) | every take is short (at most 8 words) and lexically close to a cue: best score against the list >= 0.8, where score = max of the character ratio (difflib) and the word-overlap (Jaccard) on normalised text -> keep none. One real sentence among the cues is enough to send the group to the model. List: `pipeline/step_06_llm_adjudication/keeper_cues.json` = the 26 `CUES` of `data/training/bank.py` + the takes of the natural keep-none cue group of the training folds (speaker_a_4__g1) | `cue_rule` / `RETAKE_KEEPER_CUE_RULE`; `cue_threshold` (0.8) / `RETAKE_KEEPER_CUE_THRESHOLD`; `cue_max_words` (8) | `cue_rule` |
| b. exact gate | every take has the same normalised text -> keep the last; the model is not called | `exact_gate` / `RETAKE_KEEPER_EXACT_GATE` | `exact_gate` |
| c. model | the decision model; keep = takes with p >= 0.5 (`training/decision_eval.decide`) | | `model` |
| d. empty-set fallback | the model kept nothing and the cue rule did not apply -> keep the last | `empty_fallback` / `RETAKE_KEEPER_EMPTY_FALLBACK` | `empty_fallback` |
| e. unclear | top two p within 0.15, or max p below 0.6 (`decision_eval`) -> keep the last, flag unclear for review | `unclear_default` / `RETAKE_KEEPER_UNCLEAR_DEFAULT` | `model` (status unclear) |
| model error | the model cannot answer (Studio down, timeout, bad answer) -> keep the last, flag unclear, log an ERROR | always on | `model_error_fallback` |

For `gpt-oss` the rule layers are off by default, so its behaviour is unchanged; `keeper.rules_with_gpt_oss` (`RETAKE_KEEPER_RULES_WITH_GPT_OSS=1`) turns layers a and b on in front of it.

The model input is built with the training builders, unchanged (`keeper_prompt.build_keeper_messages("short")` -> `training/build_decision_data.to_row` -> `training/build_decision_data_v2.rich_questions`), with the transcript unit just before and just after the group as context. Rebuilt this way, the `state` and `questions` of all 131 natural groups are identical to the rows the fold models were scored on.

**Decision record.** Every Step 6 span keeps its old fields (`adjudication_status`, `keeper_unit_id`, `units_to_remove`, `confidence`, `reasoning`) and gains optional ones: `keeper_status` (resolved / unclear / failed), `keeper_provider`, `keeper_model`, `decided_by`, `keeper_unit_ids` (all keepers), `keep_none`, `unclear`, `unit_probs`, `keeper_error`, and `model_keep` when the unclear rule overrode the model. `keeper_unit_id` is the last keeper (null for keep none). The service appends every advice decision to `service/data/keeper_decisions.jsonl` (provider, model, decided_by, status, keep, probabilities; marked as advice, never used as a label). `/v1/adjudicate` items gain optional `provider`, `decided_by`, `model`, `unclear`, `probs`. As before, `keep` is never empty, so a keep-none decision (cue rule) is left out of the array and only recorded in the decisions file.

## 4. Model and runtime

- Model: `runs/keeper_loop/final_all/model` (git-ignored): Qwen3-4B-Instruct-2507 in 4-bit with LoRA adapters (132 MB) and a decision head (64 MB). Loaded in Unsloth Studio as `clef-ft:Qwen_final`; a check of 12 held-out synthetic groups through the Studio API gave the same keep sets and probabilities as the in-process scoring of the run.
- Recipe (`final_all/recipe.json`): learning rate 1e-4 (head 3e-4), batch 1 x 8, 3 epochs, LoRA rank 16 / alpha 16 on all seven projections, seed 3407; data = all 131 natural groups + 1,200 synthetic; 501 steps, 95 min, peak 4.99 GB. Details: `docs/finetune_final/REPORT.md`.
- **Why the Studio API and not in-process.** The model needs Unsloth (Python 3.13, torch 2.13 with CUDA 13 in the Studio environment). The repo environment runs torch 2.5.1 (CUDA 12.4) for WhisperX and the cross-encoder; installing Unsloth there would replace torch. So `finetuned` calls `POST http://127.0.0.1:8888/v1/systemone` (`state` sent as a JSON object) with `UNSLOTH_API_KEY` from `.env`. The backend refuses any non-localhost URL. No new Python dependency was added (only `requests`, already required). Settings: `RETAKE_KEEPER_STUDIO_URL`, `RETAKE_KEEPER_MODEL`.
- **GPU.** The decision model stays resident in Studio. On the 8 GB RTX 3070 Ti, the e2e run peaked at 7,889 of 8,192 MiB with WhisperX large-v2 transcribing while Studio held the model (4,741 MiB in use before the run, including other apps). It fit, with little headroom. The pipeline now releases its own cached GPU memory before Stage 6 (`_free_cuda_cache`); it cannot free Studio's memory. If transcription runs out of memory, unload the decision model in Studio during transcription or use a smaller ASR model.
- Speed: about 0.5 to 0.7 s per group through the API once warm; the first group of the e2e run took 2.3 s (9 groups, Stage 6 total 6.8 s including the availability check).

## 5. Measured results (copied from the reports, with their limits)

All exact keeper-set match on the 131 labelled natural groups, one model per outer fold (the final model has no natural test set; folds 3 and 4 were the tuning set):

| system | correct of 131 |
|---|---|
| fine-tuned model alone | 114 |
| **hybrid policy (this provider, all layers on)** | **118** |
| rules only (cue rule + exact gate + keep-last) | 117 |
| keep-last | 117 |
| Groq gpt-oss-120b | 108 |

- Model minus Groq +4.6 points, 95% interval [-2.3, +11.5]; model minus keep-last -2.3 points, [-7.6, +3.1]. Both include zero.
- The hybrid policy was designed **after** these results were known, so its one-group lead over keep-last (118 vs 117) should not be read as an improvement.
- Policy layer counts on the 131 groups: model 104, exact gate 20, empty fallback 7, cue rule 0; 9 groups flagged unclear. Against the model alone the policy gains video_01__g15, video_07__g5, video_05__g3, video_05__g5, video_05__g6 and video_05__g8, and loses speaker_a_4__g1 (a real keep-none cue group whose takes are not all in the cue list, so the fallback forces a keeper) and video_02__g1 (a two-keeper group: two confident keepers are "within 0.15" by the unclear rule, so it keeps the last only). Source: `docs/finetune_final/data/policy_scores.json` (stored fold-model probabilities; the cue list for each fold excludes that fold's own natural cue group).
- Rules only equals keep-last on this data because the cue rule never fires on the natural groups. The lexical band (0.8) was chosen after scoring all 347 natural takes against the bank cues: at 0.8 it flags 7 takes, all real cues ("Here we go.", "Let's go."), and no real sentence; at 0.6 it would also flag a real sentence that merely ends in a cue. The one natural cue group, speaker_a_4__g1, is still not caught when scored without its own phrases ("I'm gonna start from the top." scores 0.67); in deployment its phrases are on the list. Only one natural cue group exists, so the threshold is not validated.
- One seed per fold plus one repeat on fold 4 (7 of 34 groups changed between seeds). Probabilities are saturated (almost always 0 or 1).

## 6. Files changed

| file | change |
|---|---|
| `pipeline/step_06_llm_adjudication/keeper_policy.py` (new) | KeeperConfig, cue / exact rules, model-input builder (training builders), KeeperDecision, the policy |
| `pipeline/step_06_llm_adjudication/keeper_providers.py` (new) | StudioDecisionBackend (localhost), FinetunedProvider, GptOssProvider (wraps `adjudicate_single_span` / `run_stage6`), provider choice, `run_keeper_stage` |
| `pipeline/step_06_llm_adjudication/keeper_cues.json` (new) | the cue list and its sources |
| `pipeline/run_pipeline.py` | `keeper_provider` / `keeper_config` / `keeper_backend` arguments, Stage 6 dispatch (gpt-oss without rules runs the old `run_stage6` call), `--keeper-provider` CLI flag, `groq` added to `--llm-provider`, `PipelineResult.keeper_provider`, `_free_cuda_cache` |
| `service/config.py` | `keeper_provider` (`RETAKE_KEEPER_PROVIDER`), `keeper_decisions_path` |
| `service/app.py` | capabilities `keeper` block, `keeper_provider` request field on `/v1/adjudicate`, finetuned advice path, decisions recorded |
| `service/llm.py` | `keeper_advice`, `record_decision`, cached units for context (the gpt-oss `adjudicate_groups` is unchanged) |
| `service/__main__.py` | startup log line for the keeper default and availability |
| `tests/test_keeper_providers.py` (new) | 20 tests (section 7) |
| `tests/test_service_api.py` | the flow test now pins `keeper_provider="gpt-oss"` (it tests the LLM path) and checks capabilities by key instead of whole-dict equality |
| `docs/frontend_integration/backend_overview.md` | `keeper_provider`, capabilities `keeper` block, new optional advice fields, env vars |

## 7. Tests

`pytest`: **206 passed** (186 before the change + 20 new; no GPU, no network, no key). The new tests cover each policy layer (cue rule and its lexical band, exact gate, model keep set and multi-keepers, empty-set fallback, unclear default, model-error fallback, every layer switched off), that the model input comes from the training builders (and, when `runs/keeper_loop` exists, equals the scored rows), provider precedence (default, env, request / CLI), the CLI flag, capabilities, a clear 400 for an unavailable or unknown provider, gpt-oss selected by the request field, gpt-oss output identical to the old `run_stage6` on the same mocked input, rule layers off for gpt-oss by default and on with the flag, and a no-network run of the default provider (GROQ_API_KEY unset, every non-localhost socket connection refused, the decision API faked by a local HTTP server).

## 8. End-to-end test (sample_b, cut mode `sentences`)

`runs/e2e_sample_b_hybrid_sentences/RESULT.md` (default provider) and `runs/e2e_sample_b_gptoss_sentences/RESULT.md` (gpt-oss), both git-ignored:

| | finetuned (default) | gpt-oss |
|---|---|---|
| groups | 9: model 8, empty_fallback 1 (span_3, unclear) | 9 resolved |
| removals / keep spans | 12 / 11 | 12 / 11 |
| rendered vs sum of keep spans | 28.445 s vs 28.230 s (+0.215 s) | 29.029 s vs 28.800 s (+0.229 s) |
| keeper stage | 6.80 s | 13.21 s |
| external connections | none (localhost only) | the Groq API |

The providers differ on one group, one group of three near-identical takes: finetuned keeps the last by the fallback, gpt-oss keeps the middle take. sample_b has no keeper labels, so neither is scored.

## 9. Known limitations

1. No accuracy gain over keep-last on the labelled data; the policy's 118 vs 117 was chosen after the fact.
1a. Accepted tradeoff (user decision, 2026-10-28): the unclear rule's margin also flags two confident keepers, which then keep only the last take (video_02__g1); such groups are left to the human review step.
2. The model needs Studio running with the model loaded; it cannot load in the repo environment.
3. GPU headroom is small when Studio holds the model during transcription (peak 7.9 of 8 GB on sample_b).
4. The unclear rule (from `decision_eval`) treats two confident keepers as unclear and keeps only the last, which costs multi-keeper groups (video_02__g1).
5. The cue rule needs every take to score >= 0.8 against the list; a cue group with an unlisted phrasing (scoring 0.67 against the bank cues) goes to the model, and if the model keeps nothing the fallback keeps the last take. The threshold rests on one natural cue group.
6. `/v1/adjudicate` cannot return keep-none advice (the contract says `keep` is never empty), so cue-rule groups get no advice in the editor.
7. Probabilities are saturated, so unclear flags come mostly from the empty-set fallback.
8. The pipeline writes a transcript copy to `data/raw/transcripts/<video>.json` (existing behaviour), which later runs reuse.


## 10. Update: the standalone local keeper (replaces the Studio dependency in sections 2 and 9.2-9.3)

Written for the project authors, 2026-10-28. Sections 1-9 above describe the first version, where `finetuned` called the Unsloth Studio Decision API. This section is what changed. Nothing above was deleted.

### What changed
- `finetuned` now runs the exported decision model **in the repo environment**: `decision_standalone.StandaloneDecisionModel`, mode `4bit` (the trained pre-quantized base plus the LoRA adapters). No Unsloth, no Studio, no extra server. A test scans `pipeline/`, `service/`, `models/` and `app/` for Unsloth imports.
- The Studio backend class is still in `keeper_providers.py` but is no longer the default.
- Model location: `keeper.model_path` (default `models/decision_standalone/`). If the folder is empty or missing and `keeper.model_repo` (a Hugging Face repo id) is set, the files for the chosen mode are downloaded once and verified against `export_manifest.json`. It never downloads into a folder that has files. `keeper.verify_model` checks the SHA-256 of the files in use on every load. A missing model is a clear error.
- GPU: the transcription model and the scorer's cross-encoders are released before the decision model loads; the decision model is unloaded after the keeper step (`keeper.unload_after`, default on).
- `keeper.on_model_error`: `fail_closed` (default: a group the model cannot answer fails and nothing is cut; if the model is not available at all the run stops with a clear error) or `keep_last` (keep the last take, flagged unclear, `decided_by=model_error_fallback`).
- `4bit_requant` is not offered; `keeper.mode` accepts `4bit` and `fp16` only.

### Policy layers, in order (unchanged)
1. cue rule: every take is a short production cue -> keep none (`cue_rule`)
2. exact gate: every take has the same normalised text -> keep the last (`exact_gate`)
3. model: p(yes) >= 0.5 per take is kept (`model`)
4. empty fallback: the model keeps nothing -> keep the last (`empty_fallback`)
5. unclear: top two within 0.15 or max below 0.6 -> keep the last, flagged `unclear`
Then the error policy above. There is no separate "rules" provider; `keep-last` (no model at all) still exists.

### How to switch providers
Providers `finetuned` (default), `gpt-oss` (the existing Groq call, unchanged) and `keep-last`. Precedence: request field `keeper_provider` > CLI `--keeper-provider` > `RETAKE_KEEPER_PROVIDER` > default. `PUT /v1/keeper {"provider": ...}` switches the running service. Other settings: `RETAKE_KEEPER_MODEL_PATH`, `RETAKE_KEEPER_MODEL_REPO`, `RETAKE_KEEPER_MODE`, `RETAKE_KEEPER_VERIFY_MODEL`, `RETAKE_KEEPER_UNLOAD_AFTER`, `RETAKE_KEEPER_ON_MODEL_ERROR`. `GET /v1/capabilities` has `keeper.current`, `keeper.providers.<name>.available` and, new, `keeper.finetuned` with `backend`, `mode`, `model_path`, `loaded`, `on_model_error`, `unload_after`.

### Results, stated plainly
**Held-out (earlier fold models that never saw their fold; `hybrid_comparison.md`):** model alone 114/131, keep-last 117/131, hybrid 118/131. Folds 3 and 4 were used to choose the training recipe, and the hybrid policy was designed after the fold results were known, so 118 vs 117 is not evidence of an improvement.

**In-sample:** the final model was trained on all four folds (`data_config.json`: `train_folds` 1-4, no test folds; all 131 labelled natural groups are in the training set). Its 126/131 top-keeper accuracy and the hybrid policy's 122/131 exact-set matches on those groups are **in-sample** and say nothing about new videos. The 220 synthetic groups used for parity are held out from training (no overlap).

**Parity with the Unsloth model** (`standalone_parity_report.md` for raw scores): through the full hybrid policy, final keepers agree on 131/132 natural groups (differs: `speaker_a_6__g4`) and 218/220 synthetic groups (differ: `synth_1025__g1`, `synth_1041__g1`). Deciding layer and unclear flag are identical on every natural group. Each difference is a unit near the 0.5 keep threshold (for example 0.588 vs 0.448), so the raw-score parity PASS did not rule out a rare flipped decision.

### Hardware
`4bit` (default): about 4.3 GB peak allocated during the keeper stage on sample_b, 5.4 GB in the parity run (16k-token groups), about 0.4 s per group once loaded. `fp16` needs at least 10 GB free; on an 8 GB GPU it offloads to the CPU and is about 3x slower. The VRAM figures count the PyTorch allocator only (WhisperX/CTranslate2 memory is not included).

### End-to-end sample_b, cut mode `sentences` (run folders are git-ignored)
| | finetuned (standalone, 4bit) | gpt-oss (Groq) |
|---|---|---|
| folder | `runs/e2e_sample_b_finetuned_20261028_231205` | `runs/e2e_sample_b_gptoss_20261028_231538` |
| groups | 9: model 8, empty_fallback 1 (`span_3`, unclear) | 9 resolved |
| keep spans / output length | 11 / 28.23 s | 11 / 28.80 s |
| stages (s) | transcription 32.3, scoring 7.1, keeper 157.1, cutlist 0.2, edit 2.1 (199 s total) | transcript reused, scoring 10.7, keeper 10.9, edit 2.0 |
The keeper stage of the finetuned run includes loading the model for the first time (the transcription model was released first and the GPU handoff worked). The two providers differ on one group, `span_3`: finetuned keeps unit 11:0 (fallback, flagged unclear), gpt-oss keeps 10:0. sample_b has no keeper labels, so neither is scored. The transcript for the gpt-oss run was copied from the finetuned run so unit ids match. Flagged groups of the finetuned run: `span_3` (units 9:0, 10:0, 11:0; all model probabilities near 0).

### Reproduce
`python training/standalone_export/e2e_run.py --provider finetuned --cut-mode sentences`, then `... --provider gpt-oss --reuse-transcript-from runs/e2e_sample_b_finetuned_<timestamp>` (needs `GROQ_API_KEY` in `.env`). Policy parity: `python training/standalone_export/policy_parity.py` (needs the git-ignored parity inputs from `build_parity_set.py` / `reference_scores.py`).

### Known limits that remain
`/v1/adjudicate` still cannot return keep-none advice (cue-rule groups get none); the unclear rule still costs multi-keeper groups; probabilities are saturated, so unclear flags come mostly from the empty-set fallback. The first keeper call after loading takes several seconds.
