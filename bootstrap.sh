#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_UBUNTU_VERSION="22.04"

log() {
  printf '\n[%s] %s\n' "toox3d" "$1"
}

die() {
  echo "$1" >&2
  exit 1
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

require_linux() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "This bootstrap script only supports Linux."
  fi
}

check_ubuntu() {
  if [[ ! -f /etc/os-release ]]; then
    die "Cannot detect Linux distribution. Missing /etc/os-release."
  fi

  # shellcheck disable=SC1091
  source /etc/os-release

  if [[ "${ID:-}" != "ubuntu" ]]; then
    die "This bootstrap script currently supports Ubuntu only."
  fi

  if [[ "${VERSION_ID:-}" != "$EXPECTED_UBUNTU_VERSION" ]]; then
    log "Warning: expected Ubuntu $EXPECTED_UBUNTU_VERSION, detected ${VERSION_ID:-unknown}"
  else
    log "Detected Ubuntu ${VERSION_ID}"
  fi
}

ensure_apt_packages() {
  log "Refreshing apt indexes"
  run_privileged apt-get update -y

  log "Installing base packages"
  run_privileged apt-get install -y \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-venv \
    python3-pip
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already installed"
    return
  fi

  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh

  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installation finished but uv is not in PATH."
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed"
    return
  fi

  log "Installing Docker"
  curl -fsSL https://get.docker.com | sh
  run_privileged systemctl enable --now docker
}

ensure_docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    log "Docker Compose already available"
    return
  fi

  die "Docker Compose is not available through 'docker compose'."
}

check_python() {
  log "Checking Python"
  python3 --version

  python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
PY
}

check_nvidia() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Detected NVIDIA tooling"
    nvidia-smi || true
    return
  fi

  log "nvidia-smi not found; GPU validation will need to happen later on a GPU VM"
}

check_docker() {
  log "Checking Docker"
  docker --version
  docker compose version
}

ensure_env_file() {
  if [[ -f "$PROJECT_ROOT/.env" ]]; then
    log ".env already exists"
    return
  fi

  log "Creating .env from .env.example"
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
}

ensure_directories() {
  log "Creating local directories"
  mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/scripts"
}

sync_dependencies() {
  export PATH="$HOME/.local/bin:$PATH"
  log "Syncing Python dependencies with uv"
  (cd "$PROJECT_ROOT" && uv sync)
}

print_next_steps() {
  cat <<'EOF'

Next steps:
  uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
  docker compose up --build

Health check:
  curl http://127.0.0.1:8000/health
EOF
}

main() {
  require_linux
  check_ubuntu
  ensure_apt_packages
  ensure_uv
  ensure_docker
  ensure_docker_compose
  check_docker
  check_python
  check_nvidia
  ensure_env_file
  ensure_directories
  sync_dependencies
  print_next_steps
}

main "$@"
