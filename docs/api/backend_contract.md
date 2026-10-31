# Backend handoff: the retake service

For whoever builds the Python side (the retake pipeline with the trained cross-encoder). The editor
(SimpleCut, Rust/egui) is finished on its side of the contract and tested against a fake service;
**the real HTTP client in the editor is not written yet** (see section 9). Everything here is what the
backend must provide so that client can be a thin layer.

Companion files: `docs/retake-contract.md` (short form), `docs/examples/*.json` (the exact sample
documents, tests read them), `crates/core/src/review.rs` (parser and validator: the source of truth),
`crates/core/src/analyzer.rs` (`RetakeService` trait and `FakeService`: a working model of the behaviour).

## 1. Principles

1. **The service finds and groups sentences. It never says what to keep.** No "suggested keeper",
   no scores per group, no notes. The user decides.
2. **The LLM is optional.** Not called in the normal run. Only reachable through `/v1/adjudicate`, only
   when the user presses "Ask AI to choose", and only advertised if `/v1/capabilities` says so.
3. **The editor never trusts the service.** Every document is validated (section 6). Anything wrong is
   dropped and shown to the user with a reason; a document for another file is refused.
4. **No upload.** The editor sends a local file path; the service runs on the same machine and reads the
   file itself.
5. **Stable ids.** For the same media file, sentence ids and pair ids must be identical across runs, so
   user answers can be matched again.
6. **Times are seconds (float) on the media file's own clock**, not the timeline's. The editor maps them
   to the timeline.

## 2. The flow

```
editor                                   service
  | GET  /v1/capabilities  ------------>  { adjudicate, refine }
  | POST /v1/analyze  ----------------->  job -> CANDIDATES (sentences + pairs)
  |        (user answers the ambiguous pairs, presses Next)
  | POST /v1/resolve  ----------------->  job -> GROUPS of sentences
  |        (user ticks sentences to KEEP, removes the rest)
  | POST /v1/adjudicate (optional) --->   job -> advice per group
  | POST /v1/refine (optional) ------->   job -> refined removal ranges
```

- If there is nothing to ask (no ambiguous pairs), `analyze` may answer with a **groups** document
  directly; the editor goes straight to the review.
- The editor appends the user's answers to a local `labels.jsonl` **before** sending them. The service
  should also keep them (training data: same intended line = 1, different = 0).

## 3. Endpoints

All requests and replies are JSON (UTF-8), local only (`127.0.0.1`), no auth in the first version.
Long work is a **job**: the POST returns at once with a job id, the editor polls.

### Jobs (common)

`POST` returns `202`:
```json
{ "job": "j_8f3a" }
```
`GET /v1/jobs/{job}` returns the state:
```json
{ "state": "running", "fraction": 0.42, "status": "Scoring sentence pairs" }
```
`state` is `running` | `done` | `failed` | `cancelled`. When `done` the body also has `"result"`: the
document (section 4) as a JSON object. When `failed`: `"error"`: a plain-language sentence the user can
read (it is shown verbatim: "The retake service could not finish: ..."). `fraction` is 0..1.

`DELETE /v1/jobs/{job}` cancels (idempotent, `204`).

The editor polls about every 200 ms. Keep `status` short (shown under the progress bar).

### `GET /v1/capabilities`
```json
{ "adjudicate": true, "refine": true }
```
`adjudicate: false` (or an unreachable endpoint) means the AI button cannot be used. With a `keeper` block
(below) it stays visible but disabled, showing the current keeper's reason; without one it is hidden.
Optional `keeper` block (a service that has several ways to choose which sentence to keep):
```json
{ "adjudicate": true, "refine": true,
  "keeper": { "current": "finetuned", "default": "finetuned",
              "providers": { "finetuned": { "available": true,  "reason": "decision model ..." },
                             "gpt-oss":   { "available": false, "reason": "no LLM key is configured" },
                             "keep-last": { "available": true,  "reason": "always available" } } } }
```
`adjudicate` then means "the **current** keeper can run". The editor shows a Keeper selector (unavailable
entries greyed out, `reason` as the tooltip) and, for the ids it knows (`finetuned`, `gpt-oss`, `keep-last`), a
friendly label; an unknown id is shown as sent. Ids are sent back exactly as written. Text is cut to 300
characters. `GET /v1/keeper` returns the same object as `capabilities.keeper`. `PUT /v1/keeper`
`{"provider": id}` switches the keeper for later `/v1/adjudicate` calls and returns the new state (200); an unknown or
unavailable id is a 400 `{"error": "..."}` and nothing changes. A service without a `keeper` block shows no selector.
`refine: false` (or absent) means no cut mode selector in the editor, and cuts remove whole sentence spans.
Extra fields are ignored. Example: `docs/examples/retake-capabilities.json`.

### `POST /v1/analyze`
Request (maps to `AnalysisRequest`):
```json
{
  "path": "C:/videos/take1.mp4",
  "name": "take1.mp4",
  "duration_seconds": 60.0,
  "fingerprint": "4f1c...e9f",
  "used": [ [0.0, 18.2], [24.0, 60.0] ]
}
```
- `path`: where the file is on this machine. The service reads it. If it cannot, fail the job with a
  clear message ("Could not read the file").
- `used`: the parts of the file that are on the timeline (merged, sorted, seconds). **A service may skip
  the rest**, which saves time on trimmed projects. Sentences outside `used` are simply never shown as
  actionable, so returning them is harmless.
- Result: a **candidates** document (4.1) or a **groups** document (4.3).

### `POST /v1/resolve`
Request:
```json
{
  "path": "C:/videos/take1.mp4",
  "labels": { "job": "j_8f3a", "labels": [ ... ], "skipped": [ ... ] },
  "candidates": { "...the candidates document exactly as received..." }
}
```
`labels` is the **labels request** (4.2). `candidates` is the original candidates document echoed back,
so a service that restarted (and lost the job) can still answer. The service must not require its own
memory of the earlier job.
Result: a **groups** document (4.3) with the same `job` as the candidates.

### `POST /v1/adjudicate` (optional extension)
Request:
```json
{ "path": "C:/videos/take1.mp4", "job": "j_8f3a", "groups": [ { "...groups as in 4.3, only the ones asked about..." } ] }
```
Result: **advice** (4.4). The backend implementation is Step 06 behind this endpoint; the keeper used is
`keeper.current` (see `/v1/capabilities`). The editor asks the user's consent first when sentence text can leave the
machine (the hosted `gpt-oss` keeper, or a keeper it does not know); for `finetuned` (local) and `keep-last` it does
not ask. Never use advice as training labels.

### `POST /v1/refine` (optional extension)
Request:
```json
{
  "path": "C:/videos/take1.mp4",
  "job": "j_8f3a",
  "remove": ["3:0", "4:0"],
  "mode": "anchored"
}
```
- `path`: media file path on this machine.
- `job`: id of this analysis.
- `remove`: array of sentence ids to remove. The editor sends only the removed sentences that have *not* been edited by hand (sentences whose spans the user adjusted by hand are cut exactly as set and excluded from `remove`).
- `mode`: `"anchored"` (default, "Retakes and dead air"), `"retake_only"` ("Retakes only"), or `"sentences"` ("Speech only").
Result: a job whose `done` result is **refined ranges** (4.5): `{"ranges": [{"start": 4.10, "end": 9.40}]}`.

## 4. Data models

Schema string everywhere: `"schema": "simplecut.retakes/2"`. Unknown extra fields are ignored; missing
required fields drop the entry (section 6).

### 4.1 Candidates (`"stage": "candidates"`)
Example: `docs/examples/retake-candidates.json` (14 sentences, 9 pairs, 4 ambiguous).
```json
{
  "schema": "simplecut.retakes/2",
  "stage": "candidates",
  "job": "j_8f3a",
  "media": { "path": "C:/videos/take1.mp4", "duration_seconds": 60.0 },
  "generator": { "name": "retake", "version": "0.4.0", "model": "cross-encoder-ft-v2" },
  "sentences": [
    { "id": "3:0", "start": 4.10, "end": 6.30, "text": "So now we're going to learn about", "confidence": 0.95 }
  ],
  "pairs": [
    { "id": "9:0|11:0", "a": "9:0", "b": "11:0", "class": "ambiguous", "score": 0.54 }
  ]
}
```
| Field | Type | Meaning |
|---|---|---|
| `job` | string | Id of this analysis. Echoed in every later document. |
| `media.path` / `duration_seconds` | string / number | Which file. The editor refuses the document if the length differs by more than 0.5 s (and finds the right project file by length, then by name). |
| `generator` | object | Name, version, optional `model`. Shown in logs only. |
| `sentences[].id` | string | Stable per file, e.g. `<unit>:<n>`. Unique in the document. |
| `sentences[].start/end` | number (s) | File time. `start < end`, finite, inside the file (end may exceed the length by up to 0.05 s). |
| `sentences[].text` | string | Shown to the user and used for the cards. |
| `sentences[].confidence` | number 0..1, optional | Transcription confidence. |
| `pairs[].id` | string | `<a>\|<b>`, stable. `a` is the earlier sentence. |
| `pairs[].a`, `b` | string | Sentence ids that exist in `sentences`. |
| `pairs[].class` | `"likely"` or `"ambiguous"` | Only `ambiguous` pairs become questions, shown in video order. `likely` pairs are the model's own confident edges (not asked). |
| `pairs[].score` | number 0..1 | The cross-encoder's score. **Not a calibrated probability, so the editor never shows it as a percentage**; it is for ordering and logs only. |
| `pairs[].lean` | `"retake"` or `"different"`, optional | Which way the model leans. This is what the user sees ("The model leans: same line."). Anything else drops the pair with a reason. Absent: the card shows "no lean". |
| `media.fingerprint` | string, optional | Identity of the file: SHA-256, lowercase hex, over `file size (u64, little endian) \|\| first 1 MiB \|\| last 1 MiB` (the whole file when under 2 MiB). The editor also sends it in `analyze` as `fingerprint`. The retake service currently does not send it (the editor then matches the file by duration). If present it **must** match the project's file or the document is refused; it also picks the right file when several have the same length. The empty file hashes to `af5570f5...3dfc` (pinned in a test). |
| `sentences[].internal_retake` | `{start, end}`, optional | Room for the intra-sentence restart detector. Ignored by the editor today (safe to send). |

### 4.2 Labels request (editor -> service, in `/v1/resolve`)
Example: `docs/examples/retake-labels.json`.
```json
{
  "job": "j_8f3a",
  "labels": [
    { "pair": "9:0|11:0",  "label": "retake",    "source": "user" },
    { "pair": "21:0|22:0", "label": "different", "source": "user" }
  ],
  "skipped": [ "13:0|14:0", "17:0|18:0" ]
}
```
- `label`: `"retake"` (same intended line, said again) or `"different"`. These are **ground truth for those
  pairs**: the service must not override them.
- `skipped`: pairs the user left to the model. Decide them with the cross-encoder as usual.
- `source` is always `"user"`.
- **Decided (option 1):** the user's labels are fixed edges for exactly those pairs (`retake` = score 1.0,
  `different` = 0.0, whatever the model said). The model is **not** adapted during a session and answers
  never change pairs the user did not answer. Every other pair keeps the cross-encoder's own score, then
  clustering runs over the edges. The answers are saved (`labels.jsonl`, `source: "user"`) so the model can
  be retrained offline later.

### 4.3 Groups (`"stage": "groups"`)
Example: `docs/examples/retake-groups.json` (6 groups).
```json
{
  "schema": "simplecut.retakes/2",
  "stage": "groups",
  "job": "j_8f3a",
  "media": { "path": "C:/videos/take1.mp4", "duration_seconds": 60.0 },
  "generator": { "name": "retake", "version": "0.4.0", "model": "cross-encoder-ft-v2" },
  "groups": [
    { "id": "span_0", "start": 4.10, "end": 14.60, "likely": true,
      "sentences": [
        { "id": "3:0", "start": 4.10, "end": 6.30, "text": "So now we're going to learn about" },
        { "id": "4:0", "start": 7.00, "end": 9.40, "text": "So [redacted sentence] editing" },
        { "id": "5:0", "start": 10.20, "end": 14.60, "text": "So [redacted sentence] video editing basics." }
      ] }
  ]
}
```
| Field | Meaning |
|---|---|
| `groups[].id` | Unique, stable within the job. |
| `start` / `end` | Span of the whole group = earliest sentence start, latest sentence end (10 ms tolerance). Used for the timeline highlight band and "go to". |
| `likely` | `true`: the pipeline is confident it is a retake group; the editor lists it first and expanded. `false`: a possible retake. **It is only a flag. It does not select or cut anything.** |
| `sentences[]` | At least 2, in video order. Same shape as in candidates. Ids must exist in the file's transcript; the same sentence may not be in two groups. |

Sentences that ended up in no group are not listed. **Do not include**: pairs, scores, keepers, notes.
(The editor accepts a `score` field but ignores it.)

### 4.4 Advice (adjudicate result)
Example: `docs/examples/retake-adjudication.json`.
```json
[
  { "group": "span_4", "keep": ["18:0"], "reason": "Same instruction; the second take is more specific." }
]
```
- `group`: id of a group that was sent. `keep`: ids of sentences **to keep**; all other sentences of that
  group will be unticked (as "AI" ticks, which the user can change). `reason`: short plain text, shown
  under the group. `keep` may hold **several** ids (a keeper that keeps more than one part of a group).
- Optional per item: `unclear` (true: the keeper is not sure; the group gets a "check" badge until the user
  touches its ticks, and the suggestion is still only a tick), `provider`, `decided_by`; other fields (`model`,
  `probs`) are ignored. A group missing from the array means "no suggestion" (e.g. the keeper decided "keep none").
- Rejected for that group (and noted to the user): unknown group, unknown sentence id, or an empty `keep`.
  The reply must be a plain JSON array (an object wrapper is rejected). A group answered twice uses the first answer.

### 4.5 Refined ranges (`refine` result)
```json
{
  "ranges": [
    { "start": 4.10, "end": 9.40 }
  ]
}
```
| Field | Type | Meaning |
|---|---|---|
| `ranges[].start` / `end` | number (s) | File time. `start < end`, finite, inside the file. |

### 4.6 Capabilities
`{ "adjudicate": true, "refine": true }` (section 3).

### 4.7 Local files the editor writes (read them if useful for training)
- `labels.jsonl` (in the SimpleCut data dir, `%APPDATA%\SimpleCut`): one object per user answer **before
  it is sent**:
  `{"time": <unix s>, "job", "pair", "label": "retake"|"different", "source": "user", "a": {"id","text","start","end"}, "b": {...}, "score"}`.
- `span_edits.jsonl` (see `HANDOFF.md` section 16): `{"time","job","sentence","old":{start,end},"new":{start,end},"source":"user"}`
  when a user adjusts a sentence's span. Useful as boundary-correction data.
- "Save groups..." writes the groups document with the user's edited spans.

## 5. How the frontend uses it (so you know what each field drives)

1. User presses **Remove Retakes**. The editor builds one `analyze` request per distinct video file on
   the timeline (first file only for now; it tells the user to run it again for the others).
2. **Questions step.** Ambiguous pairs, in video order, one card each: both sentences with timecodes and
   "Play"/"Go to" (play uses the timeline). Buttons: Same line, Different lines, Skip. "Next" sends the
   answers; unanswered ones are sent as `skipped` after a confirm line. The model's lean is shown ("leans: same line"), never a percentage.
3. **Review step.** Groups as colour-coded cards, `likely: true` first and expanded, `likely: false` under
   "Possible retakes" (collapsed). Each sentence has a **Keep** tick; everything starts kept, so **nothing
   is cut until the user un-ticks**. Footer: "Remove N sentences", one undo step. Helpers: Keep last only,
   Keep all, Keep last in all likely groups (user clicks, never automatic).
4. **Timeline.** Each group's `start..end` is drawn as a band; sentences that will be cut are filled,
   kept ones outlined.
5. **Live view.** The editor works out every frame which sentences are still on the timeline; a group
   with fewer than two sentences left shows as "already handled". So Undo/trim/delete done by hand
   never leave stale suggestions.
6. **Ask AI to choose** (only if `adjudicate: true`): after a consent prompt, sends the `likely: false`
   groups (all live groups if none are unsure) to `/v1/adjudicate`; ticks come back tagged "AI".
7. **Redo the questions** returns to step 2 with the earlier answers kept; **Next** calls `resolve` again
   with the new labels. So `resolve` may be called several times for one `job`; it must be repeatable.
8. A reply the editor cannot use leaves the user where they were with a message ("Your answers are
   kept."); it never half-applies.

## 6. Validation the editor applies (so you can avoid rejections)

- Documents larger than 64 MiB are refused.
- Wrong `schema` or `stage`: refused. A `/1` document is not read any more.
- `media.duration_seconds` differing from the project file by more than 0.5 s: refused ("different
  video"). `job` in a groups reply that differs from the question's job: refused.
- A sentence is dropped (with a reason shown as "N unusable entries ignored") if: empty or duplicate id,
  non-finite or reversed times, `start` < 0, `end` beyond the file by more than 0.05 s.
- A pair is dropped if either sentence is missing, `a == b`, or its `score` is not a number from 0 to 1.
- A group is dropped if: fewer than 2 valid sentences, duplicate ids, a sentence already used by another
  group, a span that does not cover its sentences (10 ms), or any non-finite value.
- Never guess: the editor does not repair, reorder across groups or fill in anything.

### Range validation for `refine` (`plan_cuts`)
When `capabilities.refine` is true and a refine reply is received, the editor checks every returned range:
- Ranges must be finite, strictly non-overlapping, and ordered (`start < end`, `start >= previous_end`).
- Each range must be at least `MIN_CUT_SECS` (0.05 s) long.
- Ranges must stay inside the media file (`end <= duration + 0.05 s`, clamped to duration).
- **Bounds check:** in `anchored` and `retake_only` modes, each range must lie within the span of removed sentences widened by `WIDEN_SECS` (1.0 s). Neighbouring widened spans merge into continuous allowed intervals.
- **The `sentences` exception:** in `sentences` mode, the "within widened spans of removed sentences" check is skipped, because `sentences` mode is designed to remove all non-speech stretches anywhere across the media. All other checks (finite, ordered, min length, file duration) still apply.
- **Fallback:** any range that violates these rules falls back to the whole sentences it touches. If any removed sentence is left uncovered by a valid range, it is cut at its full sentence span. Notes in the UI inform the user whenever a fallback occurs. Hand-edited sentences are always cut exactly as adjusted.

## 7. Errors

- Job `failed` with a human sentence in `error`. No stack traces, no file paths of the server.
- Service not running: the editor shows "The retake service could not finish: ..." and stays usable.
- HTTP 4xx on POST: body `{ "error": "..." }`, same rule.
- Never return partial documents with `done`.

## 8. Reference behaviour: the fake service

`FakeService` (`crates/core/src/analyzer.rs`) is an executable example of the above: it invents 14
sentences and 9 pairs scaled to the file's length, clusters with union-find using the user's labels, and
treats skipped ambiguous pairs with a pretend score (>= 0.8 confident, >= 0.45 unsure, else dropped);
`likely` = every edge in the group is confident. The tests in `crates/core/tests/analyzer.rs` show the
documented inputs and the groups they must produce. Mirror that behaviour (not the numbers) in the real
pipeline.

## 9. What is still to build

Backend (this document):
1. Serialise Step 04 output (units + pairs) as the candidates document, with stable ids and `class`.
2. `/v1/resolve`: user labels are final; the trained cross-encoder scores the rest; Step 05 clustering over
   surviving edges; emit groups with `start`/`end` and `likely`. Save labels with `source: "user"`.
   Decide the `likely` threshold.
3. `/v1/capabilities`; optionally `/v1/adjudicate` (Step 06 behind an endpoint, input groups, output keep lists).
4. Drop Step 06 and Step 08 (FFmpeg) from the default run; fix the `run_pipeline.py` keyword bugs first.
5. Jobs: progress, cancel, repeatable `resolve`, restart-safe via the echoed candidates.

Editor (not backend, listed so nobody assumes it exists): an HTTP implementation of the `RetakeService`
trait (`analyze`, `resolve`, `adjudicate`, `capabilities`) that speaks section 3, with timeouts, a
background poll thread per job, and cancel. Until it exists, `SIMPLECUT_FAKE_RETAKES=1` runs the fake
service, and "open analysis file" (Ctrl+Shift+R) loads a hand-made candidates or groups file, which is
also the quickest way to test your output without any HTTP at all.

## 10. How to test your output today (no HTTP needed)

1. Produce a candidates or groups JSON for a real file.
2. Open SimpleCut, import the same video, put it on the timeline.
3. Press Ctrl+Shift+R and pick your file. A candidates file opens the Questions step; a groups file opens
   the Review step.
4. Dropped entries are reported ("N unusable entries ignored"). A refusal message names the reason.
5. The same checks run in tests: `Editor::read_review` in `crates/core/src/review.rs` can be called from a
   Rust test on any document.

Endpoint paths and the job-polling shape in section 3 are a proposal made here; the editor's HTTP client
is not written, so they can still change. The JSON document shapes in section 4 are fixed and tested.
