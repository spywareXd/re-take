# ReTake backend: overview for the frontend (SimpleCut)

What the backend does, how it is built, what already exists and where, and exactly how the editor connects to it.
Companion documents: `backend-handoff.md` (the editor-side contract, schema `simplecut.retakes/2`) and `docs/frontend_integration/cut_modes.md`.
This file is the source of truth for what the **service actually does today**; where it differs from `backend-handoff.md`, the differences are listed in section 6.

## 1. The one-paragraph version

The user presses **Remove Retakes**. The editor sends the local video path to a **local HTTP service** (Python, FastAPI, `127.0.0.1:8765`). The service transcribes
the whole file, splits it into sentences, finds pairs of sentences that are probably the same line said twice, and answers with either a few **questions** for
the user (pairs the model is unsure about) or directly the **groups** of retake sentences. The user ticks which sentence in each group to keep. The service never decides
what to keep; the optional "Ask AI to choose" button is advice only. A separate `refine` call turns the sentences the user removes into exact cut ranges in file time.

```
editor                                   service (python -m service)
  | GET  /v1/capabilities  ------------>  { adjudicate, refine }
  | POST /v1/analyze  ----------------->  job -> candidates (sentences + pairs)   [or groups directly if nothing to ask]
  |        (user answers the ambiguous pairs, presses Next)
  | POST /v1/resolve  ----------------->  job -> groups (same job id)               [repeatable]
  |        (user ticks sentences to KEEP)
  | POST /v1/adjudicate (optional) --->   job -> advice per group (plain array)
  | POST /v1/refine  ------------------>  { ranges: [{start, end}] }  (sync)
```

## 2. The pipeline (what happens inside `analyze` and `resolve`)

| # | Phase | What it does | Code |
|---|---|---|---|
| 1 | Transcription | WhisperX `large-v2` on the **whole file** (never a trimmed part), word timestamps. GPU (CUDA) by default. | `pipeline/step_01_transcription/` |
| 2 | Sentences | Words are grouped into sentence-like **units**: split on pauses > 0.7 s, terminal punctuation, speaker change. A unit is a "sentence" in the documents. Start = first word start, end = last word end. | `pipeline/step_03_token_matching/unit_splitter.py` |
| 3 | Candidate pairs | Cheap lexical filter: pairs of sentences within 300 s of each other with lexical overlap >= 0.10. | `pipeline/step_03_token_matching/candidate_generator.py` |
| 4 | Scoring | Plain Stage 2 score (semantic + lexical + containment) and a fine-tuned **cross-encoder** (3 seeds) go into a **stacker**. Thresholds: confident retake (`high`), clearly not (`low`), in between = the **unsure band** (cross-encoder decides: `>= t_hi` retake, `<= t_lo` different, otherwise **ask the user**, at most 25 per video). | `pipeline/step_04_semantic_matching/`, `models/final/` |
| 5 | Groups | Union-find clustering of the confident pairs with the **user's answers as constraints**: "retake" = must-link, "different" = cannot-link. Groups of fewer than 2 sentences are dropped; a sentence is in at most one group. | `pipeline/step_05_span_aggregation/constrained_clustering.py` |
| 6 | (optional) Keeper advice | An LLM is shown a group and suggests which sentence to keep. Only on request. | `pipeline/step_06_llm_adjudication/`, `service/llm.py` |
| 7 | Review documents | Builds `candidates` and `groups` JSON, validates them exactly as the editor does, stores answers. | `pipeline/step_07_review/` |
| 8 | Cut ranges | Turns the sentences the user removes into cut ranges, in one of three cut modes (`sentences`, `anchored`, `retake_only`). | `pipeline/step_08_video_edit/cut_modes.py`, `service/refine.py` |

Steps 1-3 and the scoring in step 4 are cached per file (below), so a second run on the same file returns instantly with identical ids.

## 3. What is implemented, and where

| Capability | Status | Where |
|---|---|---|
| HTTP service (`/v1/capabilities`, `analyze`, `resolve`, `adjudicate`, `refine`, `jobs`) | done, tested | `service/` (`app.py`, `jobs.py`, `llm.py`, `refine.py`, `config.py`) |
| Job state machine (one worker thread per job, cancel, one heavy analyze at a time) | done | `service/jobs.py` |
| Candidates / groups builders | done | `pipeline/step_07_review/json_export.py` |
| Validator mirroring the editor's rules (handoff section 6) | done; every document is checked before it is sent | `pipeline/step_07_review/validator.py` |
| Media fingerprint + cache (transcript, units, pair scores) | done | `pipeline/step_07_review/cache.py` |
| Stable sentence and pair ids | done (see caveat 5.2) | same |
| Unsure band + question cap | done | `models/final/final_scorer.py`, `models/final/thresholds.json` (`unsure_band`) |
| Constrained clustering | done | `pipeline/step_05_span_aggregation/` |
| Label storage (`labels.jsonl`) | done | `pipeline/step_07_review/labels.py`, `service/data/` |
| Cut modes + ranges | done | `pipeline/step_08_video_edit/`, `docs/frontend_integration/cut_modes.md` |
| Keeper advice (LLM) | done, optional; mock LLM available | `service/llm.py` |
| Tests | `tests/test_service_api.py`, `tests/test_review_documents.py`, `tests/test_boundary_refiner.py`, `models/final/` | run `pytest tests models/final` |
| Sample documents | `service/examples/` (sample_c and sample_a: candidates + groups) and `examples/` | open with Ctrl+Shift+R |

## 4. Running the service

```
python -m service                     # 127.0.0.1:8765; options: --host, --port
```
Environment (all optional): `RETAKE_HOST`, `RETAKE_PORT`, `RETAKE_DATA_DIR` (default `service/data`), `RETAKE_LLM_PROVIDER` (`gemini` | `groq` | `openai` | `mock`),
`RETAKE_LLM_MODEL`, `RETAKE_LLM_MOCK=1` (mock keeper client, no key, no network), `RETAKE_KEEPER_PROVIDER` (`finetuned` default | `gpt-oss`; see `docs/finetune_final/pipeline_update.md`). The `finetuned` keeper needs Unsloth Studio running on 127.0.0.1:8888 with the decision model loaded and `UNSLOTH_API_KEY`. LLM keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`) are read from the environment
or the repo `.env`; `adjudicate` is advertised only if one exists or the mock is on. The startup log prints the thresholds version and what is available.

Requirements: Python env from `pyproject.toml` (`uv sync`), ffmpeg on PATH, a CUDA GPU for WhisperX (CPU works but is slow), and the trained cross-encoder weights in
`models/final/ce_seed0..2/` (**git-ignored**, you get them from the backend owner). Without them the scorer logs a WARNING and falls back to plain Stage 2: it still
works, but there is no unsure band, so no questions are asked.

**Log.** Everything the service logs also goes to `<data dir>/logs/service.log` (default `service/data/logs/service.log`, 2 MB x 5 rotated files), including full tracebacks. When the editor shows "Something went wrong while processing the file (job j_xxxx). See the service log.", search that file for the job id: `Select-String -Path service\data\logs\service.log -Pattern j_xxxx -Context 0,40`. The traceback never goes to the editor.

First analyze of a file takes roughly 50-90 s for 25-150 s of video (model loading + transcription + scoring); a repeat on the same file is instant.

## 5. The contract, as implemented

All JSON, UTF-8, local only, no auth. Errors: HTTP 4xx `{"error": "..."}`; failed jobs carry a plain sentence in `error` (no stack traces, no server paths).

**Jobs.** `POST` returns `202 {"job": "..."}`. Poll `GET /v1/jobs/{job}` about every 200 ms: `{"state": "running|done|failed|cancelled", "fraction": 0..1, "status": "Transcribing", "result"?, "error"?}`.
`DELETE /v1/jobs/{job}` cancels, always `204` (idempotent). Cancel is honoured between stages, so it can take a few seconds during transcription.

**`GET /v1/capabilities`** -> `{"adjudicate": bool, "refine": bool, "keeper": {"default": "finetuned"|"gpt-oss", "providers": {"finetuned": {"available": bool, "reason": str}, "gpt-oss": {...}}}}`. `adjudicate` is true when the default keeper provider can run.

**`POST /v1/analyze`** `{path, name?, duration_seconds?, used?: [[start, end], ...]}` -> job. Result is a **candidates** document, or a **groups** document directly if no pair needs a question.
Unreadable file -> job `failed`, error `Could not read the file`. The analyze job id **is** the document's `job`.

**`POST /v1/resolve`** `{path, labels, candidates}` -> job. `candidates` is the document **exactly as received** (see 5.1). `labels` = `{"job", "labels": [{"pair","label": "retake|different","source":"user"}], "skipped": [pair ids]}`.
Result is a groups document with the **same `job` as the candidates**; the poll job id is a new one. Repeatable: call it again after "redo the questions". The service keeps no memory of the earlier job.

**`POST /v1/adjudicate`** `{path, job, groups: [...], keeper_provider?: "finetuned"|"gpt-oss"}` -> job. Result is a **plain JSON array** `[{"group", "keep": [ids], "reason"}]`. A group the keeper fails on is omitted (never invented); `keep` is never empty, so a keep-none decision is omitted too.
With `gpt-oss` the service returns one keeper per group, as before. With `finetuned` (the default) `keep` can hold several takes, and each item adds optional `provider`, `decided_by` (`model` | `exact_gate` | `empty_fallback` | `model_error_fallback`), `model`, `unclear` (true = ask the user; the default is the last take) and `probs`.
An unknown or unavailable `keeper_provider` -> 400 with the reason; providers are never swapped silently. Every decision is appended to `service/data/keeper_decisions.jsonl`. **Advice is never used as a label.**

**`POST /v1/refine`** `{path, job, remove: [sentence ids], mode?: "anchored"|"sentences"|"retake_only"}` -> `200 {"ranges": [{"start","end"}]}` (synchronous, a few seconds). Needs the file to have been analysed
first (400 `This file has not been analysed yet...` otherwise). Ranges are in file time; see `cut_modes.md` for what each mode removes.

### Document shapes
Exactly the handoff shapes (`schema: "simplecut.retakes/2"`), plus these fields, which the editor may ignore **except** where noted:

| Where | Field | Meaning |
|---|---|---|
| candidates | `media.fingerprint` | **not sent.** The service uses a hash of the file bytes only internally, as its cache key; the editor then matches the file by duration (the field is optional in the contract) |
| candidates | `thresholds` | `high`, `low`, `t_lo`, `t_hi`, `version`, ...: **needed by `resolve`, must be echoed back** (5.1) |
| pair | `class` | `likely` (confident) or `ambiguous` (a question). Confident not-retake pairs are not emitted |
| pair | `score` | `likely`: the stacker probability. `ambiguous`: the cross-encoder probability ("the model is N% sure") |
| pair | `lean` | `ambiguous` only: `"retake"` or `"different"`, which side of the band midpoint the model leans to. Use it to pre-select a button, never to auto-answer |
| pair | `stacker_p`, `ce`, `resolved_by` | audit fields (`stacker`, `cross_encoder`, `unsure`) |
| groups | `likely` | true only if every edge in the group is confident (stacker >= high, or a user "retake"). A group with any edge decided by the cross-encoder alone is `false` |
| groups | `start`/`end` | exactly the earliest member start / latest member end (no padding) |

### 5.1 Echo the candidates document unchanged
`resolve` rebuilds the groups from the echoed document alone. The thresholds and the extra pair fields live inside it. If the editor drops unknown fields, groups still come back, but none can be marked `likely`.

### 5.2 Ids
Sentence ids look like `<segment>:<n>` and pair ids are `<a>|<b>` with `a` the earlier sentence. They are identical across runs for the same file **while the service cache exists** (`service/data/cache/<fingerprint>/`).
If that cache is deleted, re-transcription (WhisperX is not bit-deterministic) can produce different ids, and old saved answers will not match.

### 5.3 What the user is asked
At most **25** ambiguous pairs per video, those closest to the middle of the unsure band first; the rest are decided by the cross-encoder alone (and dropped). Skipped pairs are decided by `score >= t_hi`, which is always false inside the band, so a skipped question means "not a retake".

## 6. Differences from `backend-handoff.md`

1. `capabilities` also returns `refine`; `POST /v1/refine` is new (synchronous).
2. Candidates carry a top-level `thresholds` block and extra pair fields (`lean`, `stacker_p`, `ce`, `resolved_by`); candidates `media` carries **no** `fingerprint` (the editor still sends its own in `analyze`; the service ignores it).
3. The analyze job id equals the document `job`. Resolve / adjudicate jobs get their own poll ids.
4. The service itself stores the user's answers in `service/data/labels.jsonl` (skipped pairs in `labels_skipped.jsonl`), before using them, in the shape of handoff 4.6 plus `ce`, `stacker_p`, `model`, `thresholds_version`, `was_asked_because`.
5. Steps 6 (LLM) and 8 (cut/export) are **not** in the default analyze path; they exist only behind `adjudicate` and `refine`.

## 7. Limits the editor should know

- No HTTP-level auth and no upload: the path must be readable by the service's machine.
- `used` ranges are applied after scoring; the whole file is always transcribed, so trimmed projects get no speed-up.
- A restart **inside** one sentence cannot be expressed (sentences are the unit). Pairs more than 300 s apart are never considered.
- The long-video question cap (25) can drop real retakes that the model was unsure about.
- AI advice returns one sentence to keep per group.
- Only one analyze job runs at a time; later ones show status `Waiting for another analysis`.

## 8. Developing the editor without the service

Open `service/examples/sample_c_candidates.json` / `sample_c_groups.json` (or `sample_a_*`) with **Ctrl+Shift+R**; a candidates file opens the Questions step, a groups file the Review step.
For HTTP development start the service with `RETAKE_LLM_MOCK=1 python -m service` (no key needed) and call it with `curl` or your client.
The Python test `tests/test_service_api.py` is an executable example of a full session (analyze, poll, resolve, adjudicate, cancel, error).
