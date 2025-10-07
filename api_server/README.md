# Training API Server

This package exposes a lightweight HTTP service that wraps the AI Toolkit training pipeline and allows external orchestrators to control training with explicit **step budgets**. Each allocation lets the trainer advance a fixed number of steps, then it flushes checkpoints and waits for the next instruction.

## Quick start

1. Install dependencies (the root `requirements.txt` now includes FastAPI and Uvicorn).
2. Launch the server:

```bash
uvicorn api_server.app:app --host 0.0.0.0 --port 8000
```

The application exposes the following lifecycle endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | /sessions | Create a new training session from a config dictionary. |
| GET  | /sessions | List all sessions with their current status. |
| GET  | /sessions/{id} | Inspect a single session. |
| POST | /sessions/{id}/steps | Authorise an additional step budget for the active session. |
| POST | /sessions/{id}/abort | Interrupt training, force an immediate checkpoint, and release VRAM. |
| DELETE | /sessions/{id} | Abort (if still running) and remove the session. |
| GET  | /sessions/{id}/logs | Fetch recent logs. |
| GET  | /sessions/{id}/logs/stream | Stream live logs (plain-text chunked response). |

> **Note:** The legacy `/sessions/{id}/epochs` endpoint now returns HTTP 410 (Gone). Switch to `/sessions/{id}/steps` when upgrading existing clients.

### Creating a session

`config` accepts the same structure you would normally supply to `toolkit.job.get_job`. Example payload:

```json
{
  "sessionId": "my-session",
  "maxSteps": 2000,
  "config": {
    "job": "extension",
    "config": {
      "name": "my_lora_run",
      "process": [
        {
          "type": "sd_trainer",
          "training_folder": "output",
          "model": {
            "name_or_path": "black-forest-labs/FLUX.1-dev"
          },
          "datasets": [
            {
              "folder_path": "/data/dataset",
              "caption_ext": "txt"
            }
          ],
          "train": {
            "steps": 4000,
            "batch_size": 1
          }
        }
      ]
    }
  }
}
```

Once the session is created, grant step budgets as you go. For example, to run 100 steps:

```bash
curl -X POST http://localhost:8000/sessions/my-session/steps \
  -H "Content-Type: application/json" \
  -d '{ "steps": 100 }'
```

Training resumes immediately, executes the 100 authorised steps, saves a checkpoint, and pauses again. Subsequent calls can allocate new budgets (e.g. after moving the job to another GPU).

### Streaming logs

`GET /sessions/{id}/logs/stream` returns a plain-text stream (suitable for Server-Sent Events or long polling) that mirrors stdout/stderr from the underlying trainer. The stream automatically closes when training finishes or the session is deleted.

### Status payload

`GET /sessions/{id}` now includes step-aware metadata:

- `current_step` / `completed_steps` — total steps completed so far.
- `allowed_steps` — cumulative authorised budget.
- `remaining_steps` — budget that is still pending consumption.
- `last_checkpoint_step` and `last_checkpoint_path` — helpful when resuming on a different machine/GPU.

### Cleanup behaviour

- `POST /sessions/{id}/abort` signals the trainer to stop at the next safe point, forces an immediate checkpoint, and clears GPU state so the device can be reclaimed.
- `DELETE /sessions/{id}` always aborts (if needed) and drops the session from memory.

Remember to call `DELETE` or `abort` when a session is no longer needed so resources are reclaimed promptly.
