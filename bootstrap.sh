#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_UBUNTU_VERSION="22.04"
TOOX_ENV_FILE="$PROJECT_ROOT/.env"

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

wait_for_apt_lock() {
  local attempts=60
  local sleep_seconds=5
  local lock_path="/var/lib/dpkg/lock-frontend"

  for ((i=1; i<=attempts; i++)); do
    if ! run_privileged fuser "$lock_path" >/dev/null 2>&1; then
      return 0
    fi

    log "Waiting for apt/dpkg lock (${i}/${attempts})..."
    sleep "$sleep_seconds"
  done

  die "Timed out waiting for apt/dpkg lock to be released."
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
  wait_for_apt_lock

  log "Refreshing apt indexes"
  run_privileged apt-get update -y

  wait_for_apt_lock

  log "Installing base packages"
  run_privileged apt-get install -y \
    curl \
    ca-certificates \
    git \
    build-essential \
    libopengl0 \
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
  if [[ -f "$TOOX_ENV_FILE" ]]; then
    log ".env already exists"
    return
  fi

  log "Creating .env from .env.example"
  cp "$PROJECT_ROOT/.env.example" "$TOOX_ENV_FILE"
}

ensure_directories() {
  log "Creating local directories"
  mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/scripts"
}

set_env_value() {
  local key="$1"
  local value="$2"

  if grep -q "^${key}=" "$TOOX_ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$TOOX_ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >> "$TOOX_ENV_FILE"
  fi
}

get_env_value() {
  local key="$1"
  local fallback="$2"
  local value

  value="$(grep -E "^${key}=" "$TOOX_ENV_FILE" | tail -n 1 | cut -d '=' -f2- || true)"
  if [[ -z "$value" ]]; then
    printf '%s' "$fallback"
  else
    printf '%s' "$value"
  fi
}

prompt_value() {
  local prompt_text="$1"
  local default_value="$2"
  local user_value=""

  if [[ ! -t 0 ]]; then
    printf '%s' "$default_value"
    return
  fi

  read -r -p "$prompt_text [$default_value]: " user_value
  if [[ -z "$user_value" ]]; then
    printf '%s' "$default_value"
  else
    printf '%s' "$user_value"
  fi
}

configure_runtime_values() {
  local current_port current_local_port current_ssh_port current_ssh_host
  local toox_port local_tunnel_port ssh_port ssh_host

  current_port="$(get_env_value "TOOX_PORT" "8011")"
  current_local_port="$(get_env_value "TOOX_LOCAL_TUNNEL_PORT" "$current_port")"
  current_ssh_port="$(get_env_value "TOOX_SSH_PORT" "")"
  current_ssh_host="$(get_env_value "TOOX_SSH_HOST" "")"

  if [[ -t 0 ]]; then
    log "Configure server and tunnel values"
    toox_port="$(prompt_value "FastAPI port on the server" "$current_port")"
    local_tunnel_port="$(prompt_value "Local port on your Mac for the SSH tunnel" "$current_local_port")"
    ssh_port="$(prompt_value "Vast.ai SSH port (leave blank if not needed now)" "$current_ssh_port")"
    ssh_host="$(prompt_value "Vast.ai public IP or hostname (leave blank if not needed now)" "$current_ssh_host")"
  else
    toox_port="$current_port"
    local_tunnel_port="$current_local_port"
    ssh_port="$current_ssh_port"
    ssh_host="$current_ssh_host"
  fi

  set_env_value "TOOX_PORT" "$toox_port"
  set_env_value "TOOX_LOCAL_TUNNEL_PORT" "$local_tunnel_port"
  set_env_value "TOOX_SSH_PORT" "$ssh_port"
  set_env_value "TOOX_SSH_HOST" "$ssh_host"
}

sync_dependencies() {
  export PATH="$HOME/.local/bin:$PATH"
  log "Syncing Python dependencies with uv (base + hunyuan-v1)"
  (cd "$PROJECT_ROOT" && uv sync --extra hunyuan-v1)
}

ensure_python_multipart() {
  export PATH="$HOME/.local/bin:$PATH"
  log "Verifying python-multipart"

  if (cd "$PROJECT_ROOT" && uv run python -c "import multipart") >/dev/null 2>&1; then
    log "python-multipart already available"
    return
  fi

  log "Installing python-multipart"
  (cd "$PROJECT_ROOT" && uv pip install python-multipart)
}

print_next_steps() {
  local toox_port local_tunnel_port ssh_port ssh_host
  toox_port="$(get_env_value "TOOX_PORT" "8011")"
  local_tunnel_port="$(get_env_value "TOOX_LOCAL_TUNNEL_PORT" "$toox_port")"
  ssh_port="$(get_env_value "TOOX_SSH_PORT" "")"
  ssh_host="$(get_env_value "TOOX_SSH_HOST" "")"

  cat <<EOF

Next steps:
  export PATH="\$HOME/.local/bin:\$PATH"
  uv run uvicorn backend.app.main:app --host 0.0.0.0 --port ${toox_port} --reload
  docker compose up --build

Health check:
  curl http://127.0.0.1:${toox_port}/health
EOF

  if [[ -n "$ssh_port" && -n "$ssh_host" ]]; then
    cat <<EOF

SSH tunnel from your Mac:
  ssh -p ${ssh_port} root@${ssh_host} -L ${local_tunnel_port}:localhost:${toox_port}

Open in your Mac browser:
  http://127.0.0.1:${local_tunnel_port}/docs
EOF
  fi
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
  configure_runtime_values
  sync_dependencies
  ensure_python_multipart
  print_next_steps
}

main "$@"
