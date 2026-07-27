#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

printf '\n[toox3d] Installing Hunyuan V1 dependencies...\n'
(cd "$PROJECT_ROOT" && uv sync --extra hunyuan-v1)

printf '\n[toox3d] Hunyuan V1 dependencies installed.\n'
printf '[toox3d] You can now test POST /generate-v1.\n'
