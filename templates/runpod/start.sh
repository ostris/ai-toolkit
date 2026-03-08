#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app/ai-toolkit"
DATA_ROOT="${AITK_DATA_ROOT:-/workspace/ai-toolkit-data}"
HF_CACHE_ROOT="${AITK_HF_CACHE:-/workspace/.cache/huggingface}"

setup_ssh() {
  if [[ -z "${PUBLIC_KEY:-}" ]]; then
    return
  fi

  echo "Setting up SSH..."
  mkdir -p ~/.ssh
  echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys

  if [[ ! -f /etc/ssh/ssh_host_rsa_key ]]; then
    ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ""
  fi
  if [[ ! -f /etc/ssh/ssh_host_dsa_key ]]; then
    ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N ""
  fi
  if [[ ! -f /etc/ssh/ssh_host_ecdsa_key ]]; then
    ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ""
  fi
  if [[ ! -f /etc/ssh/ssh_host_ed25519_key ]]; then
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ""
  fi

  service ssh start
}

export_env_vars() {
  echo "Exporting RunPod environment variables..."
  printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F= '{ print "export " $1 "=\"" $2 "\"" }' > /etc/rp_environment || true
  if ! grep -q "rp_environment" ~/.bashrc; then
    echo "source /etc/rp_environment" >> ~/.bashrc
  fi
}

init_persistent_storage() {
  echo "Preparing persistent storage at ${DATA_ROOT}..."
  mkdir -p "${DATA_ROOT}/config" "${DATA_ROOT}/datasets" "${DATA_ROOT}/output"
  mkdir -p "${HF_CACHE_ROOT}"
  mkdir -p /root/.cache
  ln -sfn "${HF_CACHE_ROOT}" /root/.cache/huggingface

  if [[ ! -d "${DATA_ROOT}/config/examples" && -d "${APP_DIR}/config/examples" ]]; then
    mkdir -p "${DATA_ROOT}/config/examples"
    cp -a "${APP_DIR}/config/examples/." "${DATA_ROOT}/config/examples/"
  fi

  if [[ ! -f "${DATA_ROOT}/aitk_db.db" ]]; then
    touch "${DATA_ROOT}/aitk_db.db"
  fi

  rm -rf "${APP_DIR}/config" "${APP_DIR}/datasets" "${APP_DIR}/output"
  ln -sfn "${DATA_ROOT}/config" "${APP_DIR}/config"
  ln -sfn "${DATA_ROOT}/datasets" "${APP_DIR}/datasets"
  ln -sfn "${DATA_ROOT}/output" "${APP_DIR}/output"
  ln -sfn "${DATA_ROOT}/aitk_db.db" "${APP_DIR}/aitk_db.db"
}

maybe_update_repo() {
  if [[ "${AITK_UPDATE_ON_START:-0}" != "1" ]]; then
    return
  fi

  local ref="${AITK_REF:-main}"
  echo "Updating repo to ref ${ref}..."
  git -C "${APP_DIR}" fetch --all --prune
  git -C "${APP_DIR}" checkout "${ref}"
  git -C "${APP_DIR}" pull --ff-only origin "${ref}"

  python -m pip install --no-cache-dir -r "${APP_DIR}/requirements.txt"

  cd "${APP_DIR}/ui"
  npm install
  npm run build
  npm run update_db
}

start_ui() {
  cd "${APP_DIR}/ui"
  echo "Starting AI Toolkit UI on port 8675..."
  exec npm run start
}

echo "Pod started."
setup_ssh
export_env_vars
init_persistent_storage
maybe_update_repo
start_ui
