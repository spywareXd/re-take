# Retake contract `retake-editor.retakes/2`

What the editor and the retake service (the owner's Python pipeline: transcribe, split into
sentences, trained cross-encoder, grouping) say to each other. Examples are in `docs/examples/`.

**The service only finds and groups sentences. It never says what to keep and never calls an LLM.**
The user decides what to keep. An LLM is an optional extra, reached only by the "Ask AI to choose"
button.

Times are seconds on the media file's own clock. Sentence ids (`<unit>:<n>`) and pair ids
(`<a>|<b>`) are stable for the same file, so answers can be matched again later.

## Flow

1. `POST /v1/analyze` -> **candidates** (sentences and pairs). With nothing to ask, the service may
   answer with **groups** at once.
2. The editor shows the `ambiguous` pairs; the user marks each *same line* / *different lines* /
   skip, then presses **Next**. The answers are appended to a local `labels.jsonl` first.
3. `POST /v1/resolve` with the **labels** -> the cross-encoder makes **groups**.
4. The user ticks the sentences to **keep**; everything not kept is cut ("Remove"), one undo step.
5. Optional: `GET /v1/capabilities` says `{ "adjudicate": true }`; only then the button exists.
   `POST /v1/adjudicate` -> advice per group. AI advice is never sent back as training labels.

Calls are background jobs (progress, cancel).

## Documents

Every document has `schema: "retake-editor.retakes/2"`, `stage`, `job`, `media {path, duration_seconds}`
and `generator {name, version, model}`.

**candidates** (`docs/examples/retake-candidates.json`): `sentences[ {id,start,end,text,confidence?} ]`,
`pairs[ {id,a,b,class:"likely"|"ambiguous",score} ]`.

**labels request** (`retake-labels.json`): `{job, labels:[{pair,label:"retake"|"different",source:"user"}], skipped:[pair ids]}`.
Optionally the whole candidates document, so a restarted service can recover.

**groups** (`retake-groups.json`): `groups[ {id,start,end,likely,sentences[...]} ]`.
`start`/`end` cover the whole group (for the highlight); `likely: true` groups are listed first and
expanded. No pairs, no keeper, no notes, no scores.

**advice** (`retake-adjudication.json`): `[ {group, keep:[sentence ids], reason} ]`.

## Validation (never guess)

- A document for another file or another `job` is refused.
- A group is dropped, with the reason shown, if it has fewer than two sentences, duplicate or
  unknown ids, non-finite or reversed times, times outside the file, or a span that does not cover
  its sentences (10 ms tolerance).
- Advice is checked the same way: an unknown id, or a reply that keeps nothing, is ignored for that
  group and noted.
- Documents over 64 MiB are refused.

## In the editor

Every sentence has a keep tick; all start kept, so nothing is cut until the user chooses.
Users can untick all sentences in a group ("Remove all") to delete the entire section from the timeline.
A group with fewer than two sentences left on the timeline is shown as "already handled" (Undo
brings it back). Files made by hand can be opened with Ctrl+Shift+R (candidates or groups).
The old `/1` format is no longer read.

During review playback, "Skip unticked sentences when playing" jumps over cut sentences in the player.
Highlight strength for cut markers on the timeline can be adjusted between 0.1 and 0.8 (default 0.45).

## Span editing and adjustable boundaries

Sentence boundaries can be adjusted during the review before cuts are applied:
- In the sidebar: start and end fields (in seconds, 2 decimal places) with `[-]` / `[+]` 0.05 s nudge buttons.
- On the timeline: dragging the left or right edge of a sentence mark in the top video lane (5 px hit zone).
  Sentence edge dragging takes priority over clip trimming on the timeline.

Span rules enforced by core validation:
- `start < end`
- At least `MIN_CUT_SECS` (0.05 s) duration
- Stated inside the media duration `0..=media_duration`
- Clamped against neighbouring sentences in the same group (spans cannot cross adjacent sentences)
- Group `start` and `end` are dynamically updated to `min(sentence.start)` and `max(sentence.end)`
- Edits live in the active review session (`Review.groups`), not in project history. An "edited" tag and
  "Reset span" button appear on edited sentences to revert to original times (`Review.original`).

## Span edits log (`span_edits.jsonl`)

Every span adjustment (manual input, nudge, edge drag, or reset) is appended to `span_edits.jsonl` in the application data directory next to `labels.jsonl`.
Format: JSON Lines (`application/x-ndjson`), one line per edit:
```json
{"time":1714000000,"job":"fake-1","sentence":"3:0","old":{"start":4.10,"end":6.30},"new":{"start":4.25,"end":6.20},"source":"user"}
```
- `time`: Unix timestamp in seconds
- `job`: Active analysis job ID
- `sentence`: Unique sentence ID
- `old`: Previous `{start, end}` in seconds
- `new`: Updated `{start, end}` in seconds
- `source`: `"user"`

## Saved groups file

The "Save groups..." button exports the current review groups (including all user-edited spans and group bounds) to a JSON document conforming to `schema: "retake-editor.retakes/2"`, `stage: "groups"`:
```json
{
  "schema": "retake-editor.retakes/2",
  "stage": "groups",
  "job": "fake-1",
  "media": {
    "path": "C:/videos/take1.mp4",
    "duration_seconds": 60.0
  },
  "generator": {
    "name": "retake-editor",
    "version": "0.1.0"
  },
  "groups": [
    {
      "id": "group_0",
      "start": 4.25,
      "end": 9.40,
      "likely": true,
      "sentences": [
        {
          "id": "3:0",
          "start": 4.25,
          "end": 6.20,
          "text": "So now we're going to learn about",
          "confidence": 0.97
        }
      ]
    }
  ]
}
```
This document can be loaded back into Retake Editor at any time with Ctrl+Shift+R ("Open analysis...").

## Additions (optional fields)

- `pairs[].lean`: `"retake"` or `"different"`. Shown instead of a percentage (scores are not probabilities).
- `media.fingerprint`: SHA-256 hex of `file size (u64 LE) || first 1 MiB || last 1 MiB`; must match the project file when present.
- `sentences[].internal_retake`: `{start, end}`, ignored for now.
