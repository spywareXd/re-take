# Service

A local FastAPI service that runs the pipeline for the Retake Editor editor. It listens on `127.0.0.1` only and keeps its state on disk.

```
uv run python -m service [--host 127.0.0.1] [--port 8765]
```

Configuration is by environment variables or a `.env` file in the repository root (values are never printed or logged):

| Variable | Meaning | Default |
|---|---|---|
| `RETAKE_HOST`, `RETAKE_PORT` | where to listen | `127.0.0.1`, `8765` |
| `RETAKE_DATA_DIR` | cache, labels and logs when no project folder is used | `service/data` |
| `RETAKE_KEEPER_PROVIDER` | `finetuned` (local model plus rules), `gpt-oss` (hosted), `keep-last` | `finetuned` |
| `RETAKE_LLM_MOCK` | `1` uses a fake language model (tests) | off |
| `RETAKE_OFFLINE` | `1` stops the scorer from downloading weights | off |
| `GEMINI_API_KEY`, `GROQ_API_KEY` | only for hosted keeper or LLM options | unset |

## API

Long jobs return `202 {"job": "..."}`; poll `GET /v1/jobs/{job}` or follow the progress events at `/v1/jobs/{job}/events` (polling) and `/v1/jobs/{job}/events/stream` (server-sent events).

| Endpoint | Purpose |
|---|---|
| `GET /v1/capabilities` | what this service can do (keeper available, model list, limits) |
| `POST /v1/analyze` | transcribe, split, score pairs; returns candidates and the pairs to ask |
| `POST /v1/resolve` | apply the user's answers and return the groups |
| `POST /v1/adjudicate`, `GET/PUT /v1/keeper` | optional keeper suggestion, keeper selection |
| `POST /v1/refine` | turn the sentences to remove into cut ranges for a cut mode |
| `GET/PUT /v1/project/settings`, `GET /v1/project/status`, `GET /v1/project/units-preview`, `POST /v1/project/resplit` | project folder settings, stale marks, silence-gap preview and re-split |
| `POST /v1/work/reset`, `POST /v1/work/restore` | set transcription or keeper results aside, bring them back (nothing is deleted) |
| `GET /v1/export/options`, presets, `POST /v1/export` | export presets, render and verify the output |

The full contract (every field, status codes, how the editor uses each) is in [../docs/api/backend_contract.md](../docs/api/backend_contract.md); a shorter form is [../docs/api/retake_contract.md](../docs/api/retake_contract.md), and the cut modes are in [../docs/api/cut_modes.md](../docs/api/cut_modes.md). Sample documents the editor tests read are in `../retake-editor/docs/examples/`.

## Project folders

When the editor passes a `project`, the service writes only inside it (`X/data/`). Without one it uses the per-machine cache in `RETAKE_DATA_DIR`. Replacing a result (a different model, a different silence gap) is always confirmed first (HTTP 409), and the old result is kept as a numbered copy.

## Tests

`uv run pytest tests/test_service_api.py tests/test_project_store.py tests/test_export.py` (offline; the transcriber and scorer are test doubles).
