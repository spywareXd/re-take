# App changes: models, progress, silence gap, exports, project folders

Branch `app-ux-and-storage`. The "app" is the SimpleCut editor (`simplecut/`, Rust/egui) plus the Python retake service (`service/`). The service does the work and keeps
its data; the editor shows the settings, asks before replacing anything, and sends the choices. Plan: `plan.md`. Storage audit: `storage_audit.md`.

## What changed, per item

| # | Item | Service (Python) | Editor (Rust) |
|---|---|---|---|
| 1 | Model size selector | `pipeline/step_01_transcription/model_catalog.py` builds the list from the installed faster-whisper (`tiny, base, small, medium, large-v2, large-v3, turbo`); `GET /v1/capabilities` has `transcription`; `model` on `analyze`; saved in the project (`data/settings.json`); changing it after a transcript exists marks it stale and refuses (409) until confirmed | Options window: dropdown with download size, GPU memory, speed, downloaded or not, warning |
| 2 | Progress logs | `service/events.py`: events, SSE and polling endpoints, persisted `data/progress.jsonl`; stages emitted by the transcriber, splitter, scorer, keeper, cut-list and export code | Retakes panel: stepper (all 11 stages, in order), current stage + elapsed, scrolling log, "cached", exact errors, log kept after the run |
| 3 | Silence gap | `silence_gap_s` is a job parameter, range-checked on the server; `units-preview`, `resplit`; stale marks and numbered copies | Slider + number box, live sentence count, "Re-split sentences", the confirmation |
| 4 | Exports | `service/export_options.py`, `service/export_media.py`, `POST /v1/export`, presets, optional `destination` folder | One Export dialog (see "Second round") |
| 5 | dB display overflow | | `timeline.rs` track headers (see below) |
| 6 | Remove Retakes button | | `toolbar.rs` primary button |
| 7 | Project-local storage | `service/project_store.py` (the one owner of paths under `X/data/`), legacy import | Save As makes `X/X.<ext>` + `data/`; project status and source warning on open |

### Item 5: the cause
`track_headers` laid the name, mute button and dB box left to right in a cell with 146 px of usable width; they need about 172 px ("Dialogue" is longer than "Music"),
and nothing shrank, wrapped or clipped. Fix: the controls are anchored to the right edge, the name takes what is left (shortened with "…", full text as hover and accessible
name), the cell clips, and the header column is 196 px instead of 168 (without that, "Dialogue" was cut to "Dia…"). Tests: `crates/ui/tests/timeline_headers.rs` (fails before
the fix, at 960×600, 1100×700, 1300×820 and 1920×1080 and with dB values from −60 to +12). Screenshots: `screenshots/item5_*`.

### Item 6: the button
Same size and font as the other buttons, filled with the theme's accent colour (it was first made larger; that was changed back after use). Dark label on the fill (about
5.7:1; WCAG AA needs 4.5:1, and white text would be 3.2:1). Hover lighter, pressed slightly darker (kept above 4.5:1; a first version at 4.06:1 was caught by the contrast
test), keyboard focus has a 2.5 px inside outline, disabled is flat grey with a border, while the service works the label gets the percent and the button fills like a bar
(no animation of its own; the stage is in the tooltip). Constants: `HOVER_LIGHTEN`, `PRESSED_DARKEN` in `toolbar.rs`. Current screenshots: `screenshots/item6b_*`.

## Project folder layout

```
X/                      the project folder (X = the project name)
  X.fcproj              the project file (FilmCraft's own format, written by the editor)
  FilmCraft Previews/   made by the FilmCraft engine next to the project file (inside X/)
  data/
    project.json        source video: path relative to X/ + size and sha256 fingerprint; stale marks
    settings.json       transcription_model, silence_gap_s
    transcript.json, words.json, units.json, pairs__<model>.json, keepers.json, cutlist.json,
    transcript_meta.json (the model it was made with), units_meta.json (the gap it was made with)
    labels.jsonl, labels_skipped.jsonl, keeper_decisions.jsonl     answers and decisions (append-only)
    labels_editor.jsonl, span_edits.jsonl                          the editor's own logs for this project
    progress.jsonl      the event log (item 2)
    audio/              extracted audio
    cache/              other cached intermediates
    stale/              numbered copies of replaced results: units.1.json, transcript.2.json, ...
    exports/            exports: unique names, never overwritten
    pipeline/           (a full pipeline run, `run_full_pipeline(processed_dir=...)`: candidates, cutlists, units, outputs)
```
Every JSON object is `{"schema_version": 1, "kind": ..., "data": ...}`, written atomically (temp file in the same folder, then rename). A file without the envelope loads as
schema 0; a newer schema is refused with a plain sentence. The source video is referenced, not copied; reopening checks the fingerprint and warns once if it is missing or changed.
**Legacy:** nothing was deleted or moved. The old per-machine cache is read as a fallback when no project is open; with a project open the window offers "Import into this project",
which **copies** (`POST /v1/project/import-legacy`) and leaves the originals untouched.

## Event schema

`{seq, job, stage, status, pct?, message, timestamp, cached?}`

| field | meaning |
|---|---|
| `seq` | 1, 2, 3, … per job (resume with `?after=<seq>` or the SSE `Last-Event-ID`) |
| `stage` | `model_download, model_load, audio_extraction, vad, transcription, alignment, unit_splitting, pair_scoring, keeper_selection, cutlist_build, export_render` |
| `status` | `started`, `progress`, `done`, `error` |
| `pct` | 0 to 100, when known (download bytes, transcription batches, ffmpeg `-progress`) |
| `message` | for `error`: `<ExceptionType>: <the exact message>` (ffmpeg's own stderr for a render error). Local paths can appear in it; the job's own `error` stays a sanitised sentence |
| `cached` | `true` with `status: done` and message `cached`: the stage was skipped because its result already existed |

Endpoints: `GET /v1/jobs/{id}/events?after=N` (JSON), `GET /v1/jobs/{id}/events/stream` (SSE; ends with `event: end`), `GET /v1/project/events?project=<folder>` (the persisted log).
The editor polls (its HTTP client is blocking); SSE is there for other clients. Emitting never blocks on a listener; a test runs a 1.5 s job while 30 requests are answered
without a wait over 0.5 s.

## Export options

| Option | Choices | Notes |
|---|---|---|
| Resolution | match source, 2160p, 1440p, 1080p, 720p, 480p, vertical 1080×1920, square 1080×1080, custom W×H | height presets keep the source aspect; custom sizes must be even and 16 to 7680; aspect lock, or exact size with letterbox / crop / stretch; upscaling is noted |
| Frame rate | match source, 24, 25, 30, 50, 60, custom (1 to 240) | match source keeps a variable frame rate; the result says CFR or VFR (checked with ffprobe on the written file) |
| Codec | H.264, H.265 | NVENC offered only if ffmpeg lists the encoder **and** a test encode works |
| Quality | CRF presets draft / standard / high (x264 28 / 23 / 18, x265 32 / 28 / 23), explicit CRF 0 to 51, or bitrate 100 to 200000 kbps | encoder speed ultrafast to veryslow |
| Audio | AAC 64 to 512 kbps; crossfade at cuts 0 to 100 ms (default 15 ms = `CutConfig.fade_ms`, the pipeline's own setting); loudness normalisation off by default | the cut uses the pipeline's own `build_concat_filter` |
| Presets | Match Source, YouTube 1080p, YouTube 4K, Shorts / Reels / TikTok, Draft / Preview, and your own | yours are saved in the user config (`%APPDATA%\SimpleCut\export_presets.json`, or `RETAKE_USER_DIR`), not in the project |
| Other exports | audio only (MP3, WAV), subtitles of the kept text (SRT, VTT, timings on the cut timeline), cut list (EDL, JSON), transcript (TXT, JSON) | |

Assumptions (Filmcraft's built-in presets are documented in `filmcraft/crates/export/src/presets.rs`): the list follows its "Match Source", "Web & Social" and "Audio Only" groups where
they overlap with the list above. ProRes / DNxHR / MXF, image sequences and GIF stay in the editor's own Filmcraft export and are not offered here. The YouTube presets use
CRF instead of Filmcraft's fixed 16 / 45 Mbps. EDL timecode is non-drop-frame at the nearest whole rate (29.97 is written as 30 and the file says so).

## How to change defaults

| Default | Where |
|---|---|
| Transcription model (`large-v2`) | `DEFAULT_MODEL` in `pipeline/step_01_transcription/model_catalog.py` (and `ReviewConfig.transcription` in `pipeline/step_07_review/config.py`) |
| Model sizes, VRAM, speed shown | `APPROX` in `model_catalog.py` (approximate, static) |
| Compute type order | `COMPUTE_TYPES_GPU` / `COMPUTE_TYPES_CPU` in `model_catalog.py` |
| Silence gap (0.7 s) and its range | `PAUSE_SPLIT_THRESHOLD_SECONDS` in `pipeline/step_03_token_matching/config.py`; `SILENCE_MIN_S / MAX_S / STEP_S` in `service/app.py` |
| Export presets, CRF values, option ranges | `BUILTIN_PRESETS`, `CRF_PRESETS` and the range constants in `service/export_options.py` |
| Crossfade at cuts (15 ms) | `CutConfig.fade_ms` in `pipeline/step_08_video_edit/cut_config.py` |
| Button colours / sizes | `HOVER_LIGHTEN`, `PRESSED_DARKEN`, `PRIMARY_W`, `PRIMARY_H` in `simplecut/crates/ui/src/toolbar.rs`; the accent is `theme::DARK.accent` |

## Known limits

- **Screenshots** are drawn by a small software renderer written for the tests (`crates/ui/tests/common/raster.rs`): layout-faithful, not pixel-identical to the GPU window. A real-window capture was not possible (the GL surface comes out black).
- **Model sizes, VRAM and speed** are approximate (a static table), labelled so; the real size on disk is shown once a model is cached.
- **Alignment model download** has no byte progress (WhisperX gives no hook); the stage shows started and done. VAD and transcription percent come from WhisperX's own progress lines.
- **VFR detection** looks at the first 400 packets and at `r_frame_rate` vs `avg_frame_rate`.
- **Editor against the real service** is not tested end to end: the editor tests use scripted service doubles, the client is tested against a mock HTTP server, and the service is tested through its API. The export options fixture is a capture of the real service's reply.
- **Without a project folder** (an unsaved or old project) the service falls back to its old locations, and the extracted audio is written next to the video as before.
- **Still outside the project folder, by design:** the service's own log (`service/data/logs/service.log`), the editor's crash-recovery autosave (`%APPDATA%\SimpleCut`), the Hugging Face and torch model caches, and the user's export presets.
- **`max_absorb_s`** of the cut modes still imports the 0.7 s constant; changing the silence gap does not change it.
- **Cold `GET /v1/capabilities`** can take tens of seconds the first time (the keeper availability check loads its libraries). That check is the keeper's and was not changed.
- The existing core test `export_to_unwritable_path_fails_cleanly` assumes the drive `X:` does not exist; on a machine with an `X:` drive it exports a real file there (it did here: `X:\cannot\exist\rendered.mp4`). It is skipped in the runs for this work and was not changed.
- Items 1, 3 and 7 share files in the editor and were committed together (`3e2172f`).

## Second round (after first use)

| Reported | Cause | Change |
|---|---|---|
| Hovering the transcription dropdown shifts the other rows | egui's hovered button grows its layout a little, and the rows below follow | The dropdown rows (Options and Export) are drawn by `menu_row`: a fixed-size slot with a hand-painted highlight. Test: `hovering_over_the_model_dropdown_and_its_items_moves_no_row` (fails on the old code) |
| Remove Retakes button too big | the first restyle made it larger | Same size and font as the other buttons, still blue |
| Ask AI (local Qwen keeper) keeps every sentence at a 0.1 s gap | **Not the model.** `ask_ai` sent only the "unsure" groups when any existed, so "likely" groups were never asked and stayed fully ticked. A shorter gap makes a mix of likely and unsure groups; at 0.7 s all groups were likely, so all were asked. On `sample_a` at 0.1 s: 5 groups, 4 likely, 1 unsure; only the unsure one was sent. Run directly with the real keeper, every group got advice with some sentences removed | A local keeper (nothing leaves the machine) is asked about **every** group. The hosted keeper keeps the narrower, consent-gated behaviour. Tests: `a_local_keeper_is_asked_about_every_group_not_only_the_unsure_ones`, `a_hosted_keeper_is_still_asked_...` |
| Remove / Apply button hidden below the cut modes | The footer (three unwrapped cut-mode buttons, wrapped description and warning, the button) was laid out under a *guessed* reserved height, so it overflowed the panel | The footer is laid out from the bottom up, the cut modes wrap, and the list of groups takes the rest. Test: `the_remove_button_and_the_cut_modes_stay_inside_the_panel_...` at four window sizes and 1.3x / 1.5x text (it fails on the old layout: bottom 794 vs 792) |
| Ctrl+Z should undo a step's files; a Reset option | Steps that made files were not part of undo, so a redo with new settings found the old results | See "Ctrl+Z and Reset" |
| One Export menu, with destination folder and Export button inside | There were two Export buttons and dialogs | See "One Export dialog" |
| Saving after the cache exists: the transcription should follow | The first analysis lives in the service's old per-machine cache, and the saved project started empty | Right after the first save the editor asks the service to **copy** that work into the project (`POST /v1/project/import-legacy` with the video path). Originals stay. The same call backs the Options button |

Also fixed on the way: the old per-machine cache kept one sentence split per video and ignored the silence gap (a shorter gap silently did nothing). It now records the gap
that made the split (`units_meta.json`) and splits again, reusing the transcript, when a different gap is asked for.

### Ctrl+Z and Reset
- A step that **created files** (the transcription: transcript, words, sentences, pair scores; the AI's choices) is recorded like an edit. **Ctrl+Z** undoes the latest one if
  nothing was edited after it (otherwise the edit comes first, as usual); **Ctrl+Y** redoes it. A step that found everything in the cache made nothing and is not recorded.
- "Undo" **sets the work aside, it never deletes**: the service marks the results as reset (`POST /v1/work/reset`), the files stay, and the next run computes them again
  without asking (the old ones become `stale/<name>.<n>.json`). `POST /v1/work/restore` (Ctrl+Y) lifts the mark. The user's answers (`labels.jsonl`) are never touched.
  It works for a project folder and for the old per-machine cache (`<fingerprint>/_reset.json`).
- Undoing the transcription closes the review (it belonged to the old transcription); undoing the AI puts the ticks back as they were.
- Options window, "Start over": **Reset transcription…** and **Reset keeper groups…** (each asks first). Doing something new after an undo ends the redo.
- Not covered: the cut list file (`cutlist.json`) is replaced with a numbered copy when a cut is applied again, but it has no step of its own in Ctrl+Z.

### One Export dialog
One **Export** button (Ctrl+E) opens "Export options". With the retake service connected it starts with **What to export**: *Video — whole timeline (MP4)* (the editor's own
render: picture, dialogue, music, volume; its own destination file with Browse and the "replace it?" question) and the service's kinds (cut video with all the options,
audio only, subtitles, cut list, transcript), which go to a **destination folder** (default `data/exports`; Browse folder…; never overwrites, names get `_2`, `_3`).
A saved project with the service connected opens on the service video; otherwise on the whole timeline. The Export button ("Start export") is inside the dialog.
