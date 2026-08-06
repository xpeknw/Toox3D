# Toox 3D

Toox 3D is a self-hosted application for AI-driven 3D asset generation.

This initial milestone focuses on a reproducible Ubuntu 22.04 GPU development
environment with:

- `uv` for Python environment and dependency management
- FastAPI for the API
- Docker and Docker Compose for containerized development
- a minimal `/health` endpoint

The lockfile will be generated from a real dependency resolution on the target
machine the first time `uv sync` runs.

## Current scope

This repository does not migrate Hunyuan yet.

The current goal is to make a fresh GPU VM capable of running:

```bash
./bootstrap.sh
docker compose up --build
```

and then exposing:

```text
GET /health
```

After the infrastructure milestone, the repository also includes a first
compatibility endpoint:

```text
POST /generate-v1
```

This endpoint is intended to reproduce the working Hunyuan V1 behavior as
closely as practical outside Colab.

The first server boot may take several minutes because it may need to:

- install `uv`
- install Docker
- resolve Python dependencies, including the Hunyuan V1 stack
- clone the Tencent Hunyuan3D-2 repository
- download Hunyuan model weights from Hugging Face
- initialize the Hunyuan pipeline once in GPU memory
- build the Docker image
- start `uvicorn` automatically in the background

During `bootstrap.sh`, the script can ask for:

- the FastAPI port on the server
- the local port you want to use on your Mac
- the Vast.ai SSH port
- the server public IP or hostname

Those values are stored in `.env` and reused later.

You can also pass them directly to avoid interactive prompts:

```bash
./bootstrap.sh \
  --port 8011 \
  --ssh-port 27608 \
  --ssh-host 151.237.25.234 \
  --install-trellis
```

If you pass any of those values, bootstrap skips the interactive questions and
fills the rest from defaults or previously saved `.env` values. `--local-port`
defaults to the same value as `--port`.

For fully non-interactive runs without passing any values:

```bash
./bootstrap.sh --no-prompt
```

The bootstrap script also:

- persists `export PATH="$HOME/.local/bin:$PATH"` into `~/.bashrc` and `~/.profile`
- checks `nvidia-smi` early before expensive work
- saves a failed NVIDIA state into `/tmp/toox3d_nvidia_retry_state`
- reboots once automatically if NVIDIA is unavailable on first boot
- aborts on the next run if NVIDIA is still unavailable, so you can destroy the instance
- predownloads the background-removal model (`u2net.onnx`)
- preloads the Hunyuan repo and weights on GPU hosts
- optionally installs TRELLIS in `models/trellis-venv`
- optionally preloads TRELLIS weights when `TOOX_INSTALL_TRELLIS=1`
- starts `uvicorn` automatically on the configured port
- writes the API log to `logs/uvicorn.log`

If you want TRELLIS available in the engine selector, bootstrap must install it:

```bash
./bootstrap.sh --install-trellis
```

That keeps TRELLIS isolated from the main Hunyuan runtime by using a separate
virtual environment at `models/trellis-venv`.

TRELLIS uses a heavier dependency stack than Hunyuan. The bootstrap currently
installs its isolated runtime incrementally around the official project
requirements, including `open3d`, `kaolin`, and `xformers`. The isolated worker
prefers `ATTN_BACKEND=xformers` with `SPCONV_ALGO=native` to avoid relying on
`flash-attn` during initial bring-up. Depending on GPU/CUDA/driver
combinations, additional native extensions from the upstream TRELLIS setup may
still be required later.

## Local development

Requirements:

- Python 3.10+
- `uv`

Install dependencies:

```bash
uv sync
```

This creates `uv.lock` if it does not exist yet.

This sync installs both the API dependencies and the Hunyuan V1 stack.

Run the API:

```bash
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Docker development

Build and start:

```bash
docker compose up --build
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

Stop:

```bash
docker compose down
```

## Generate V1

After a successful GPU bootstrap, the Hunyuan repo and model weights should
already be downloaded and the first request penalty should be much smaller.

Example request:

```bash
curl -X POST "http://127.0.0.1:8011/generate-v1" \
  -F "image=@/absolute/path/to/input.png" \
  -F "preset=v1-stable"
```

Download bundle ZIP with `metadata.json + STL`:

```bash
curl -L "http://127.0.0.1:8011/downloads/<job-id>/bundle" \
  --output modelo_bundle.zip
```

Download full ZIP with `metadata.json + STL + OBJ + GLB + processed image`:

```bash
curl -L "http://127.0.0.1:8011/downloads/<job-id>/all" \
  --output modelo_all.zip
```

Direct STL download:

```bash
curl -L "http://127.0.0.1:8011/artifacts/<job-id>/STL/<file>.stl" \
  --output modelo.stl
```

Expected response shape:

```json
{
  "job_id": "part_20260727_ab12cd34",
  "exports": {
    "stl": {"ok": true, "path": "..."},
    "obj": {"ok": true, "path": "..."},
    "glb": {"ok": true, "path": "..."}
  },
  "download_urls": {
    "stl": "/artifacts/<job-id>/STL/<file>.stl",
    "obj": "/artifacts/<job-id>/OBJ/<file>.obj",
    "glb": "/artifacts/<job-id>/GLB/<file>.glb",
    "processed_image": "/artifacts/<job-id>/processed_image/imagen_procesada.png"
  },
  "bundle_urls": {
    "bundle": "/downloads/<job-id>/bundle",
    "all": "/downloads/<job-id>/all"
  }
}
```

## V2 Jobs

V2 is designed so the heavy generation work finishes independently of the
download. Submit a job first, poll its status, and download only when it is
ready.

Create a job:

```bash
curl -X POST "http://127.0.0.1:8011/v2/jobs" \
  -F "image=@/absolute/path/to/input.png" \
  -F "preset=high"
```

List available presets:

```bash
curl "http://127.0.0.1:8011/v2/presets"
```

Typical response:

```json
{
  "job_id": "pieza_20260727_123456_ab12cd34",
  "status": "queued",
  "progress_percent": 0,
  "progress_message": "Queued",
  "effective_params": {
    "preset": "high",
    "octree_resolution": 512,
    "num_inference_steps": 30,
    "guidance_scale": 5.5,
    "seed": 1234,
    "remove_background": true
  },
  "status_url": "/v2/jobs/pieza_20260727_123456_ab12cd34",
  "timing": {
    "average_completed_job_seconds": 120.0,
    "timing_basis_preset": "high",
    "queue_position": 1,
    "estimated_wait_seconds": 0.0,
    "estimated_total_seconds": 120.0
  },
  "bundle_urls": {
    "bundle": "/downloads/pieza_20260727_123456_ab12cd34/bundle",
    "all": "/downloads/pieza_20260727_123456_ab12cd34/all"
  }
}
```

Completed jobs also include a `result_summary` with practical output metrics such
as STL/OBJ/GLB size, bundle size, vertices, faces, and actual processing time.

Built-in presets:

- `v1-stable`: `384 / 30 / 5.5`
- `high`: `512 / 30 / 5.5`
- `max`: `512 / 40 / 5.5`

Manual overrides can still be sent together with a preset. For example, you can
submit `preset=high` and override only `num_inference_steps=36`.

Poll a single job:

```bash
curl "http://127.0.0.1:8011/v2/jobs/<job-id>"
```

List recent jobs:

```bash
curl "http://127.0.0.1:8011/v2/jobs?limit=20"
```

Download when the job reaches `completed`:

```bash
curl -L "http://127.0.0.1:8011/downloads/<job-id>/bundle" \
  --output modelo_bundle.zip
```

Cancel a queued job:

```bash
curl -X POST "http://127.0.0.1:8011/v2/jobs/<job-id>/cancel"
```

Delete a completed, failed, or cancelled job and its outputs:

```bash
curl -X DELETE "http://127.0.0.1:8011/v2/jobs/<job-id>"
```

Cleanup old jobs in bulk:

```bash
curl -X POST "http://127.0.0.1:8011/v2/jobs/cleanup" \
  -F "older_than_hours=24" \
  -F "statuses=completed,failed,cancelled"
```

## Fresh Ubuntu 22.04 VM

For the current validation phase, the repository is assumed to be publicly
readable and cloned over HTTPS.

Target flow:

```bash
git clone https://github.com/xpeknw/Toox3D.git
cd Toox3D
./bootstrap.sh
docker compose up --build
```

Suggested verification sequence on the VM:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8011 --reload
curl http://127.0.0.1:8011/health
docker compose up --build
curl http://127.0.0.1:8011/health
docker compose ps
```

Example SSH tunnel from your Mac after bootstrap is configured:

```bash
ssh -p 32145 root@123.123.123.123 -L 8011:localhost:8011
```

Example from a real Vast.ai test:

```bash
ssh -p 27608 root@151.237.25.234 -L 8011:localhost:8011
```

## Notes

- `models/` and `outputs/` are intentionally kept out of Git.
- Hunyuan integration comes after infrastructure, Docker, and GPU validation.
- Private repository authentication is intentionally out of scope for this
  milestone. If the repository becomes private later, authentication should be
  handled as a separate pre-bootstrap step.
