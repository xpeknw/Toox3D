ROADMAP.md

Toox 3D Roadmap

This roadmap describes the next work from the exact point where infrastructure testing stopped.

The previous Vast.ai VM was intentionally destroyed.

The next environment will begin from a fresh Ubuntu GPU server.

Phase 0 — Infrastructure proof of concept

Status: completed manually.

Validated:

Vast.ai Ubuntu VM rental

Ubuntu 22.04

RTX 3090 with 24 GB VRAM

SSH access

Root access

Python 3.10.12

Docker

Docker Compose

FastAPI

Uvicorn

SSH port forwarding

Swagger UI from the local Mac browser

Result:

The proposed remote development workflow is viable.

No files from the destroyed server should be assumed to exist.

Phase 1 — Create the repository foundation

Status: next task.

Create the initial repository:

toox-3d/
├── backend/
│   └── app/
│       ├── __init__.py
│       └── main.py
├── scripts/
├── models/
├── outputs/
├── .env.example
├── .gitignore
├── AGENTS.md
├── ROADMAP.md
├── README.md
├── bootstrap.sh
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── uv.lock

Tasks:

Initialize Git.

Initialize the Python project with uv.

Add FastAPI.

Add Uvicorn.

Add configuration support only if immediately needed.

Create a basic /health endpoint.

Add .gitignore.

Add .env.example.

Add startup instructions to README.md.

Acceptance criteria:

uv sync
uv run uvicorn backend.app.main:app --reload

starts the API successfully.

Phase 2 — Build the first bootstrap installer

Status: pending.

Create an idempotent:

bootstrap.sh

Initial responsibilities:

Confirm Linux environment.

Refresh apt indexes.

Install essential base packages.

Install uv if missing.

Confirm Python compatibility.

Confirm nvidia-smi.

Confirm Docker.

Install Docker only when missing.

Confirm Docker Compose.

Create required directories.

Create .env from .env.example only when missing.

Install project dependencies.

Print useful status and next steps.

Do not download Hunyuan weights in the first bootstrap version.

Acceptance criteria:

A fresh compatible Ubuntu VM can prepare the basic project using one command:

./bootstrap.sh

Running the script twice does not break the environment.

Phase 3 — Containerize the health service

Status: pending.

Tasks:

Create a minimal Dockerfile.

Create docker-compose.yml.

Mount the source during development if useful.

Expose the FastAPI port.

Add a container health check.

Confirm logs are readable.

Add a restart policy only if appropriate.

Acceptance criteria:

docker compose up --build

starts Toox 3D and:

GET /health

returns:

{"status": "ok"}

Phase 4 — Verify NVIDIA GPU access in Docker

Status: pending.

Tasks:

Verify nvidia-smi on the host.

Verify Docker can use --gpus all.

Install or configure NVIDIA Container Toolkit only if required.

Select a CUDA/PyTorch image compatible with the future Hunyuan environment.

Add a reproducible GPU validation command or script.

Do not install the full host CUDA Toolkit without a specific dependency requiring it.

Acceptance criteria:

A container launched by the project can detect the RTX 3090 and report its VRAM.

Phase 5 — Recreate the environment on a second clean VM

Status: pending.

Purpose:

Prove that the installer is real and not accidentally dependent on the first server.

Tasks:

Destroy or ignore the first development VM.

Rent a fresh compatible GPU VM.

Clone the repository.

Run ./bootstrap.sh.

Run docker compose up.

Test /health.

Test GPU visibility.

Acceptance criteria:

No undocumented manual fix is necessary.

Any required manual step must be added to the installer or README before this phase is considered complete.

Phase 6 — Analyze the existing Colab notebook

Status: pending.

Input:

The approximately 835-line Google Colab notebook that already generates 3D assets and STL files with Hunyuan3D-2.

Tasks:

Identify system dependencies.

Identify Python dependencies.

Identify model repositories and checkpoints.

Identify environment variables.

Identify model initialization.

Identify image preprocessing.

Identify mesh generation.

Identify mesh cleanup.

Identify STL, OBJ or GLB export.

Identify notebook-only behavior that must be removed.

Document the pipeline before moving code.

Acceptance criteria:

A migration checklist exists and every notebook dependency has a target location in the normal application.

Phase 7 — Migrate Hunyuan model loading

Status: pending.

Tasks:

Add the required Python dependencies.

Add system dependencies to Docker or bootstrap.

Add model configuration.

Add a model download or preparation script.

Load Hunyuan once when the worker or API starts.

Log startup progress.

Handle missing weights clearly.

Confirm GPU memory usage.

Avoid loading the model for every request.

Acceptance criteria:

The model initializes successfully on the RTX 3090 without relying on Colab-specific code.

Phase 8 — Migrate one synchronous generation flow

Status: pending.

Keep this phase deliberately simple.

Tasks:

Accept one image.

Run the migrated Hunyuan pipeline.

Save one generated 3D output.

Export the format already proven by the notebook.

Return useful metadata or a download path.

Add error handling.

Keep only one generation active at a time if required.

Possible first endpoint:

POST /generate

Acceptance criteria:

An uploaded image produces a valid 3D file using the same essential logic that worked in Colab.

Do not add Redis, Celery or a frontend before this works.

Phase 9 — Persistent model process and job state

Status: pending.

Tasks:

Keep the model resident in GPU memory.

Prevent accidental duplicate model initialization.

Introduce a simple job identifier.

Track queued, running, completed and failed status.

Add:

GET /jobs/{job_id}

This phase may still use in-memory state for initial testing.

Acceptance criteria:

Multiple API requests do not reload the model each time.

Phase 10 — Asynchronous generation queue

Status: future.

Only begin after synchronous generation is stable.

Evaluate:

A small internal worker

Redis Queue

Celery

Another lightweight queue

Choose based on actual needs rather than adding infrastructure preemptively.

Acceptance criteria:

The API returns quickly with a job ID and generation runs outside the request lifecycle.

Phase 11 — Web interface

Status: future.

Possible functionality:

Upload or generate an input image

Choose generation settings

View progress

Preview the 3D model

Download STL, OBJ or GLB

Send the output to a later slicing or printing workflow

The user is considering Ionic for the application interface.

Do not begin this phase until the backend generation API is stable.

Phase 12 — Additional providers

Status: future.

Possible providers:

TRELLIS

Stable Fast 3D

Tripo

Other future image-to-3D models

Introduce a common provider interface only after the first Hunyuan implementation establishes the real shared requirements.

Current next action

The next Codex task should be limited to:

Create the initial repository structure.

Add a minimal FastAPI /health endpoint.

Configure the project with uv.

Create the first idempotent bootstrap.sh.

Create a minimal Dockerfile and Docker Compose configuration.

Document exactly how to test it on a fresh Vast.ai Ubuntu VM.

Do not proceed to Hunyuan integration until these items are complete and tested.