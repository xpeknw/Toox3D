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

The first server boot may take several minutes because it may need to:

- install `uv`
- install Docker
- resolve Python dependencies
- build the Docker image

During `bootstrap.sh`, the script can ask for:

- the FastAPI port on the server
- the local port you want to use on your Mac
- the Vast.ai SSH port
- the server public IP or hostname

Those values are stored in `.env` and reused later.

## Local development

Requirements:

- Python 3.10+
- `uv`

Install dependencies:

```bash
uv sync
```

This creates `uv.lock` if it does not exist yet.

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
