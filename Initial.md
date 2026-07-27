AGENTS.md

Toox 3D

Purpose

Toox 3D is a self-hosted application for generating 3D assets with artificial intelligence.

The project started as a Google Colab notebook using Tencent Hunyuan3D-2. The notebook already proved that the generation pipeline can create usable 3D models and STL files from images.

The project is now being migrated from Colab into a permanent, reproducible Linux application.

The current goal is not to improve the model pipeline yet. The immediate goal is to build the infrastructure and installer required to recreate the complete development environment on a disposable GPU server.

Current project state

The original implementation exists as a Google Colab notebook of approximately 835 lines.

That notebook currently contains the working generation logic and should be treated as the reference implementation for the first Hunyuan provider.

The new application has not yet been created as a complete repository.

Infrastructure testing was performed manually on a Vast.ai Ubuntu VM.

The test server was destroyed after validation, so the next server will start from zero.

Infrastructure already validated

The following workflow has already been tested successfully:

MacBook
   ↓
SSH
   ↓
Ubuntu 22.04 GPU VM on Vast.ai
   ↓
FastAPI + Uvicorn
   ↓
SSH tunnel
   ↓
Swagger UI in the local browser

The tested machine had:

Ubuntu 22.04

NVIDIA RTX 3090

24 GB VRAM

NVIDIA driver 580

CUDA compatibility reported by nvidia-smi

Python 3.10.12

Docker 28

Docker Compose 2.35

SSH access

Root access

A minimal FastAPI application was started successfully.

Swagger was opened locally through an SSH tunnel at:

http://localhost:8080/docs

This validated the development architecture.

Important decisions already made

Google Colab is no longer the primary environment

Colab became unreliable because GPU sessions were interrupted and available GPU time was inconsistent.

Colab may remain useful for experiments, but Toox 3D must run as a normal Linux application.

Vast.ai servers are disposable

Keeping the tested VM stopped would have cost approximately:

$0.87 USD per day

Because stopped storage is not cheap enough to justify preserving every development VM, the server was destroyed.

The project must therefore be fully reproducible.

Code must survive; servers do not need to survive

The repository, configuration, scripts and documentation are the permanent assets.

A GPU VM should be replaceable at any time.

Do not depend on manual installation

Any command needed to prepare a fresh server must eventually be placed in scripts or configuration files.

The desired installation flow is:

git clone https://github.com/xpeknw/Toox3D.git
cd Toox3D
./bootstrap.sh

The script should prepare the server without requiring the developer to remember the previous manual steps.

For the current validation phase, the repository may remain public so a fresh
server can clone it over HTTPS without SSH keys, GitHub CLI authentication or
tokens.

If the repository becomes private later, repository authentication must be
handled as a separate step before running bootstrap.sh.

For Vast.ai development, bootstrap.sh may ask for connectivity values such as:

- the FastAPI port on the server
- the local tunnel port on the Mac
- the Vast.ai SSH port
- the server IP or hostname

These values may be written to .env so the script can print the exact SSH
tunnel command for the current machine.

Example SSH tunnel used successfully during testing:

ssh -p 27608 root@151.237.25.234 \
    -L 8011:localhost:8011

Immediate objective

Build the first reproducible Toox 3D repository and installer.

The next work should focus on:

Creating the repository structure.

Creating bootstrap.sh.

Creating the Python project with uv.

Creating a minimal FastAPI service.

Creating Docker and Docker Compose configuration.

Verifying NVIDIA GPU access.

Preparing model storage and output directories.

Documenting how to start the development server.

Only after the environment is reproducible, migrate the Hunyuan logic from the Colab notebook.

Do not start with frontend development.

Do not start by redesigning the generation pipeline.

Do not add authentication, payments, user management or unrelated services.

Finish the reproducible foundation first.

Project name

The project is called:

Toox 3D

Suggested repository directory:

toox-3d

Do not call the complete application Hunyuan Studio.

Hunyuan is the first model implementation, not the product name.

Proposed initial repository structure

Keep the first version simple.

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

Do not create unnecessary abstractions or empty enterprise-style folders before they are needed.

The structure may grow later when the actual Hunyuan integration is migrated.

Python tooling

Use:

uv

for:

Project initialization

Virtual environment management

Dependency installation

Lock file generation

Running Python commands

Preferred basic commands:

uv sync
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

The tested server had Python 3.10.12.

The first version should remain compatible with Python 3.10 unless the Hunyuan dependencies require another supported version.

FastAPI requirements

The initial API only needs a minimal health endpoint.

Example:

from fastapi import FastAPI

app = FastAPI(title="Toox 3D")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

The objective is to verify installation and networking, not to build all final endpoints immediately.

Likely future endpoints include:

GET  /health
POST /generate
GET  /jobs/{job_id}
GET  /jobs/{job_id}/download

Do not implement the future endpoints until the base environment is working.

Bootstrap requirements

bootstrap.sh is the immediate priority.

It should eventually:

Exit on errors.

Detect that it is running on Linux.

Detect or document the expected Ubuntu version.

Update apt package indexes.

Install required base packages.

Install uv if missing.

Install Docker if missing.

Install Docker Compose if missing.

Verify that docker works.

Verify that nvidia-smi works.

Verify or configure Docker GPU support.

Create required local directories.

Copy .env.example to .env when .env does not exist.

Run uv sync when using the host development environment.

Build or pull the required Docker image.

Print the exact next command to start Toox 3D.

The script must be idempotent.

Running it more than once should not destroy configuration or duplicate installations.

Avoid interactive prompts where possible.

Do not download huge model weights until the base installer is verified.

Model download may later be controlled by a separate command or environment variable.

Docker requirements

Docker and Docker Compose were already available on the tested Vast.ai Ubuntu template.

However, the installer must not assume every future server has the same template.

The initial Docker configuration should run the FastAPI application.

GPU support must eventually be tested from inside the container.

A useful validation command is conceptually equivalent to:

docker run --rm --gpus all <cuda-compatible-image> nvidia-smi

Choose a suitable CUDA/PyTorch base image only after checking compatibility with the Hunyuan dependencies.

Do not install Ubuntu's full CUDA Toolkit just because nvcc is missing.

The tested machine had a working NVIDIA driver and nvidia-smi, even though nvcc was not installed.

For PyTorch-based inference, the host driver plus the CUDA runtime shipped in the container or Python packages may be sufficient.

Model storage

Model weights must not be committed to Git.

Suggested configurable directories:

./models
./outputs

or mounted server directories such as:

/srv/toox-3d/models
/srv/toox-3d/outputs

The exact path should be configurable through environment variables.

Example variables:

TOOX_MODELS_DIR=/srv/toox-3d/models
TOOX_OUTPUTS_DIR=/srv/toox-3d/outputs

The first implementation may use local project directories for simplicity.

Generated output

Generated files should be kept outside the source code.

A possible future layout is:

outputs/
└── <job-id>/
    ├── input.png
    ├── preview.png
    ├── model.glb
    ├── model.obj
    └── model.stl

Do not commit generated files.

Hunyuan integration

Hunyuan3D-2 is the first generation engine because the Colab notebook already uses it successfully.

The notebook should be migrated only after:

The repository exists.

bootstrap.sh works.

FastAPI starts reliably.

Docker can access the GPU.

The environment can be recreated from a fresh VM.

When migrating:

Extract reusable generation logic from notebook cells.

Do not preserve notebook-specific state.

Keep the model loaded between generation requests.

Avoid reloading model weights for every request.

Separate model initialization from request handling.

Preserve working generation behavior before refactoring aggressively.

Add logging around model loading, generation, export and failures.

The first goal is parity with the existing notebook, not a total rewrite.

Future provider design

Toox 3D should not be permanently tied to Hunyuan.

Other providers may be added later, such as TRELLIS, Stable Fast 3D, Tripo or other models.

However, do not build an elaborate provider framework before the first Hunyuan implementation works outside Colab.

A small interface may be introduced when it provides immediate value.

Avoid speculative abstractions.

Development workflow

Preferred workflow:

MacBook
   ↓
VS Code Remote SSH
   ↓
Edit files directly on the Ubuntu VM
   ↓
FastAPI runs with --reload
   ↓
Test from local browser through SSH tunnel

The developer should not need to upload files manually after every code change.

Git should be used from the beginning so the server can be destroyed safely.

Use tmux when a long-running process must survive an SSH disconnect during development.

Coding behavior for Codex

When working on this repository:

Complete the requested task before proposing a larger redesign.

Do not replace working code without a concrete reason.

Do not create unrelated features.

Do not add frontend code until requested.

Do not add authentication until requested.

Do not add queues until synchronous generation works.

Keep scripts readable and commented only where useful.

Prefer a small working implementation over a large speculative architecture.

Report commands that were actually tested.

Clearly identify commands that still need to be tested on a GPU server.

Do not claim Hunyuan works until an actual model generation succeeds.

Preserve environment configurability.

Never commit secrets, model weights, generated outputs or virtual environments.

Definition of done for the current milestone

The current milestone is complete when a fresh supported Ubuntu GPU VM can run:

git clone https://github.com/xpeknw/Toox3D.git
cd Toox3D
./bootstrap.sh
docker compose up

and then successfully expose:

GET /health

with Docker able to see the NVIDIA GPU.

Hunyuan model integration comes after this milestone. 
