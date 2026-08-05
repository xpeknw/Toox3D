#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_UBUNTU_VERSION="22.04"
TOOX_ENV_FILE="$PROJECT_ROOT/.env"
TOOX_LOG_DIR="$PROJECT_ROOT/logs"
TOOX_UVICORN_LOG="$TOOX_LOG_DIR/uvicorn.log"
TOOX_UVICORN_PID_FILE="$TOOX_LOG_DIR/uvicorn.pid"
TOOX_NVIDIA_RETRY_FILE="/tmp/toox3d_nvidia_retry_state"
CLI_TOOX_PORT=""
CLI_LOCAL_TUNNEL_PORT=""
CLI_SSH_PORT=""
CLI_SSH_HOST=""
CLI_INSTALL_TRELLIS=""
CLI_NO_PROMPT="0"

log() {
  printf '\n[%s] %s\n' "toox3d" "$1"
}

die() {
  echo "$1" >&2
  exit 1
}

print_usage() {
  cat <<EOF
Usage:
  ./bootstrap.sh [options]

Options:
  --port <value>          FastAPI port on the server
  --local-port <value>    Local SSH tunnel port on your Mac (defaults to --port)
  --ssh-port <value>      Vast.ai SSH port
  --ssh-host <value>      Vast.ai public IP or hostname
  --install-trellis       Install TRELLIS runtime during bootstrap
  --no-prompt             Do not ask interactive questions
  --help                  Show this help
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        [[ $# -ge 2 ]] || die "--port requires a value"
        CLI_TOOX_PORT="$2"
        shift 2
        ;;
      --local-port)
        [[ $# -ge 2 ]] || die "--local-port requires a value"
        CLI_LOCAL_TUNNEL_PORT="$2"
        shift 2
        ;;
      --ssh-port)
        [[ $# -ge 2 ]] || die "--ssh-port requires a value"
        CLI_SSH_PORT="$2"
        shift 2
        ;;
      --ssh-host)
        [[ $# -ge 2 ]] || die "--ssh-host requires a value"
        CLI_SSH_HOST="$2"
        shift 2
        ;;
      --install-trellis)
        CLI_INSTALL_TRELLIS="1"
        shift
        ;;
      --no-prompt)
        CLI_NO_PROMPT="1"
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        die "Unknown option: $1. Run ./bootstrap.sh --help"
        ;;
    esac
  done
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
  local lock_paths=(
    "/var/lib/dpkg/lock-frontend"
    "/var/lib/apt/lists/lock"
    "/var/cache/apt/archives/lock"
  )
  local all_free="1"

  for ((i=1; i<=attempts; i++)); do
    all_free="1"
    for lock_path in "${lock_paths[@]}"; do
      if run_privileged fuser "$lock_path" >/dev/null 2>&1; then
        all_free="0"
        break
      fi
    done

    if [[ "$all_free" == "1" ]]; then
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
  local diagnostics=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    diagnostics="$(nvidia-smi 2>&1 || true)"
    if nvidia-smi >/dev/null 2>&1; then
      log "Detected NVIDIA tooling"
      printf '%s\n' "$diagnostics"
      rm -f "$TOOX_NVIDIA_RETRY_FILE"
      return
    fi
  else
    diagnostics="nvidia-smi not found in PATH"
  fi

  if [[ -f "$TOOX_NVIDIA_RETRY_FILE" ]]; then
    local previous_state
    previous_state="$(cat "$TOOX_NVIDIA_RETRY_FILE" 2>/dev/null || true)"
    die "NVIDIA still failed after one reboot attempt. Destroy this instance instead of spending more time or money. Previous state: ${previous_state}. Current state: ${diagnostics}"
  fi

  printf 'timestamp=%s\nstate=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$diagnostics" > "$TOOX_NVIDIA_RETRY_FILE"
  log "NVIDIA check failed. Saved state to $TOOX_NVIDIA_RETRY_FILE and rebooting once."
  run_privileged reboot
  exit 0
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
  mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/scripts" "$TOOX_LOG_DIR"
}

ensure_local_bin_on_path() {
  local export_line='export PATH="$HOME/.local/bin:$PATH"'

  export PATH="$HOME/.local/bin:$PATH"

  for shell_file in "$HOME/.bashrc" "$HOME/.profile"; do
    touch "$shell_file"
    if ! grep -Fq "$export_line" "$shell_file"; then
      printf '\n%s\n' "$export_line" >> "$shell_file"
      log "Added ~/.local/bin PATH export to $(basename "$shell_file")"
    fi
  done
}

ensure_toox3d_command() {
  local command_path="$HOME/.local/bin/toox3d"
  local alias_line="alias toox3d='uv run uvicorn backend.app.main:app --host 0.0.0.0 --port \$(grep -E \"^TOOX_PORT=\" \"$TOOX_ENV_FILE\" | tail -n 1 | cut -d '=' -f2- || echo 8011) --reload'"

  mkdir -p "$HOME/.local/bin"

  cat > "$command_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$PROJECT_ROOT"
TOOX_ENV_FILE="$TOOX_ENV_FILE"
export PATH="\$HOME/.local/bin:\$PATH"

TOOX_PORT="\$(grep -E '^TOOX_PORT=' "\$TOOX_ENV_FILE" | tail -n 1 | cut -d '=' -f2- || true)"
if [[ -z "\$TOOX_PORT" ]]; then
  TOOX_PORT="8011"
fi

cd "\$PROJECT_ROOT"
exec uv run uvicorn backend.app.main:app --host 0.0.0.0 --port "\$TOOX_PORT" --reload
EOF

  chmod +x "$command_path"

  for shell_file in "$HOME/.bashrc" "$HOME/.profile"; do
    touch "$shell_file"
    if ! grep -Fq "alias toox3d=" "$shell_file"; then
      printf '\n%s\n' "$alias_line" >> "$shell_file"
      log "Added toox3d alias to $(basename "$shell_file")"
    fi
  done
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
  local current_port current_local_port current_ssh_port current_ssh_host current_install_trellis
  local toox_port local_tunnel_port ssh_port ssh_host install_trellis
  local cli_values_supplied="0"

  current_port="$(get_env_value "TOOX_PORT" "8011")"
  current_local_port="$(get_env_value "TOOX_LOCAL_TUNNEL_PORT" "$current_port")"
  current_ssh_port="$(get_env_value "TOOX_SSH_PORT" "")"
  current_ssh_host="$(get_env_value "TOOX_SSH_HOST" "")"
  current_install_trellis="$(get_env_value "TOOX_INSTALL_TRELLIS" "0")"

  toox_port="${CLI_TOOX_PORT:-}"
  local_tunnel_port="${CLI_LOCAL_TUNNEL_PORT:-}"
  ssh_port="${CLI_SSH_PORT:-}"
  ssh_host="${CLI_SSH_HOST:-}"
  install_trellis="${CLI_INSTALL_TRELLIS:-}"

  if [[ -n "$toox_port" || -n "$local_tunnel_port" || -n "$ssh_port" || -n "$ssh_host" || -n "$install_trellis" ]]; then
    cli_values_supplied="1"
  fi

  if [[ -n "$toox_port" && -z "$local_tunnel_port" ]]; then
    local_tunnel_port="$toox_port"
  fi

  if [[ "$CLI_NO_PROMPT" == "1" || "$cli_values_supplied" == "1" ]]; then
    [[ -n "$toox_port" ]] || toox_port="$current_port"
    [[ -n "$local_tunnel_port" ]] || local_tunnel_port="$toox_port"
    [[ -n "$ssh_port" ]] || ssh_port="$current_ssh_port"
    [[ -n "$ssh_host" ]] || ssh_host="$current_ssh_host"
    [[ -n "$install_trellis" ]] || install_trellis="$current_install_trellis"
  elif [[ -t 0 ]]; then
    log "Configure server and tunnel values"
    [[ -n "$toox_port" ]] || toox_port="$(prompt_value "FastAPI port on the server" "$current_port")"
    [[ -n "$local_tunnel_port" ]] || local_tunnel_port="$(prompt_value "Local port on your Mac for the SSH tunnel" "$toox_port")"
    [[ -n "$ssh_port" ]] || ssh_port="$(prompt_value "Vast.ai SSH port (leave blank if not needed now)" "$current_ssh_port")"
    [[ -n "$ssh_host" ]] || ssh_host="$(prompt_value "Vast.ai public IP or hostname (leave blank if not needed now)" "$current_ssh_host")"
    [[ -n "$install_trellis" ]] || install_trellis="$(prompt_value "Install TRELLIS runtime too? (0/1)" "$current_install_trellis")"
  else
    [[ -n "$toox_port" ]] || toox_port="$current_port"
    [[ -n "$local_tunnel_port" ]] || local_tunnel_port="$toox_port"
    [[ -n "$ssh_port" ]] || ssh_port="$current_ssh_port"
    [[ -n "$ssh_host" ]] || ssh_host="$current_ssh_host"
    [[ -n "$install_trellis" ]] || install_trellis="$current_install_trellis"
  fi

  set_env_value "TOOX_PORT" "$toox_port"
  set_env_value "TOOX_LOCAL_TUNNEL_PORT" "$local_tunnel_port"
  set_env_value "TOOX_SSH_PORT" "$ssh_port"
  set_env_value "TOOX_SSH_HOST" "$ssh_host"
  set_env_value "TOOX_INSTALL_TRELLIS" "$install_trellis"
  set_env_value "TOOX_TRELLIS_VENV" "./models/trellis-venv"
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

preload_hunyuan_model() {
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "Skipping Hunyuan preload because nvidia-smi is not available"
    return
  fi

  log "Preloading Hunyuan repository and model weights"
  (
    cd "$PROJECT_ROOT" && uv run python - <<'PY'
from backend.app.services.hunyuan_v1 import service

service._ensure_background_remover()
service._ensure_pipeline()
print("[toox3d] Background remover assets and Hunyuan pipeline loaded and cached.")
PY
  )
}

install_trellis_runtime() {
  export PATH="$HOME/.local/bin:$PATH"
  local install_trellis
  local trellis_repo
  local trellis_venv
  local trellis_python

  install_trellis="$(get_env_value "TOOX_INSTALL_TRELLIS" "0")"
  if [[ "$install_trellis" != "1" ]]; then
    log "Skipping TRELLIS runtime install"
    return
  fi

  trellis_repo="$PROJECT_ROOT/models/trellis-repo"
  trellis_venv="$PROJECT_ROOT/models/trellis-venv"
  trellis_python="$trellis_venv/bin/python"

  if [[ ! -d "$trellis_repo/.git" ]]; then
    log "Cloning TRELLIS repository"
    mkdir -p "$PROJECT_ROOT/models"
    git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git "$trellis_repo"
  else
    log "TRELLIS repository already present"
  fi

  if [[ ! -x "$trellis_python" ]]; then
    log "Creating TRELLIS venv"
    python3 -m venv "$trellis_venv"
  fi

  log "Installing TRELLIS runtime dependencies"
  "$trellis_python" -m pip install --upgrade pip setuptools wheel
  "$trellis_python" -m pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
  "$trellis_python" -m pip install \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers \
    safetensors einops accelerate huggingface_hub omegaconf
  "$trellis_python" -m pip install \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

  if ! "$trellis_python" -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124; then
    log "Warning: xformers install failed for TRELLIS. Continuing without it."
  fi
}

preload_trellis_model() {
  export PATH="$HOME/.local/bin:$PATH"
  local install_trellis
  local trellis_venv
  local trellis_python

  install_trellis="$(get_env_value "TOOX_INSTALL_TRELLIS" "0")"
  if [[ "$install_trellis" != "1" ]]; then
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "Skipping TRELLIS preload because nvidia-smi is not available"
    return
  fi

  trellis_venv="$PROJECT_ROOT/models/trellis-venv"
  trellis_python="$trellis_venv/bin/python"
  if [[ ! -x "$trellis_python" ]]; then
    log "Skipping TRELLIS preload because its venv is missing"
    return
  fi

  log "Preloading TRELLIS repository and model weights"
  (
    cd "$PROJECT_ROOT"
    TOOX_TRELLIS_VENV="./models/trellis-venv" \
    uv run python - <<'PY'
from backend.app.services.trellis_v1 import service

service._ensure_pipeline()
print("[toox3d] TRELLIS pipeline loaded and cached.")
PY
  )
}

start_uvicorn() {
  export PATH="$HOME/.local/bin:$PATH"
  local toox_port
  toox_port="$(get_env_value "TOOX_PORT" "8011")"

  if [[ -f "$TOOX_UVICORN_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$TOOX_UVICORN_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
      log "Stopping existing uvicorn process ($existing_pid)"
      kill "$existing_pid" >/dev/null 2>&1 || true
      sleep 1
    fi
    rm -f "$TOOX_UVICORN_PID_FILE"
  fi

  log "Starting uvicorn on port ${toox_port}"
  (
    cd "$PROJECT_ROOT"
    nohup uv run uvicorn backend.app.main:app --host 0.0.0.0 --port "$toox_port" --reload \
      >"$TOOX_UVICORN_LOG" 2>&1 &
    echo $! > "$TOOX_UVICORN_PID_FILE"
  )

  sleep 3

  if [[ -f "$TOOX_UVICORN_PID_FILE" ]]; then
    local new_pid
    new_pid="$(cat "$TOOX_UVICORN_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$new_pid" ]] && kill -0 "$new_pid" >/dev/null 2>&1; then
      log "uvicorn started in background with PID ${new_pid}"
      return
    fi
  fi

  die "uvicorn did not stay running. Check $TOOX_UVICORN_LOG"
}

print_next_steps() {
  local toox_port local_tunnel_port ssh_port ssh_host
  toox_port="$(get_env_value "TOOX_PORT" "8011")"
  local_tunnel_port="$(get_env_value "TOOX_LOCAL_TUNNEL_PORT" "$toox_port")"
  ssh_port="$(get_env_value "TOOX_SSH_PORT" "")"
  ssh_host="$(get_env_value "TOOX_SSH_HOST" "")"

  cat <<EOF

Next steps:
  docker compose up --build

Health check:
  curl http://127.0.0.1:${toox_port}/health

Uvicorn log:
  tail -f ${TOOX_UVICORN_LOG}
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
  parse_args "$@"
  require_linux
  check_ubuntu
  check_nvidia
  ensure_apt_packages
  ensure_uv
  ensure_docker
  ensure_docker_compose
  check_docker
  check_python
  ensure_env_file
  ensure_directories
  ensure_local_bin_on_path
  configure_runtime_values
  ensure_toox3d_command
  sync_dependencies
  ensure_python_multipart
  install_trellis_runtime
  preload_hunyuan_model
  preload_trellis_model
  start_uvicorn
  print_next_steps
}

main "$@"
