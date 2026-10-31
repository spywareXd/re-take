# Cut mode (frontend contract)

**Over HTTP** the mode is the field `mode` of `POST /v1/refine` (`{path, job, remove: [sentence ids], mode}`), default `anchored`. The response is only `{"ranges": [{"start","end"}]}` in file time: there is no `removed_seconds` and no `mode_preview` over HTTP, and a `cut_mode` key in the request is ignored (the default is used). `cut_mode` / `preview_cut_modes` and the cutlist fields below belong to the Python pipeline and CLI only. The editor sends `mode`, defaults to `anchored`, and checks returned ranges itself (for `sentences` ranges can lie far from the removed sentences, by design).

`cut_mode` chooses how the resolved retake decisions become the edited video. The retake detection is identical in all modes; only the step that turns the removals into the final keep spans differs.

| field | type | default | where |
|---|---|---|---|
| `cut_mode` | enum `sentences` \| `anchored` \| `retake_only` | `anchored` | pipeline input: `run_full_pipeline(cut_mode=...)`, CLI `--cut-mode` |
| `preview_cut_modes` | bool (optional) | `false` | pipeline input; when true the cutlist gets `mode_preview` |

| value | one line for the UI |
|---|---|
| `sentences` | Keeps only spoken sentences and remove silences |
| `anchored` | Removes retakes and the silence/noise around them. Recommended. |
| `retake_only` | Removes only the retake itself. Least intrusive. |

No warning is shown for any mode.

## Output fields (cutlist JSON, additive; older cutlists without them still load)

```json
{
  "mode": "anchored",
  "removed_seconds": 33.51,
  "mode_preview": { "sentences": 105.0, "anchored": 33.51, "retake_only": 25.06 }
}
```

`mode_preview` is present only when `preview_cut_modes=true`. Existing fields (`keep_spans`, `discard_spans`, `retake_groups`, `output_duration`, `time_saved`) are unchanged.

A per-run review file `keep_spans_vs_units.csv` is written next to the cutlist (one row per keep span with its lead/trail seconds, the kept units inside, how each join stopped, and `removed_flag` rows for removed stretches over 2 s with no speech and visible motion = possible lost footage).

## Behaviour details

- `anchored` extends each removal outward through dead air (no ASR word, audio >= 13 dB below the neighbouring speech, and little visual motion), at most `max_absorb_s` per side. `max_absorb_s` is the same 0.7 s gap gate the unit builder uses (`PAUSE_SPLIT_THRESHOLD_SECONDS`), so a pause longer than that next to a removal is only partly absorbed. It keeps 0.20 s of padding next to the kept words, snaps every cut to the quietest frame within +/- 50 ms and fades the audio 15 ms at each join.
- `sentences` keeps the kept words (+ 0.20 s padding) and shortens any gap between them above 0.5 s to 0.5 s.
- `retake_only` removes just the retaken words.
- Motion threshold (`motion_threshold`, 0.25) was calibrated on static test footage (max 0.17). Footage with camera noise or gameplay needs a higher value or `use_visual_motion=False`; all values live in `CutConfig`.
