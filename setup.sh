#!/usr/bin/env bash
# Bootstrap CS336 assignments on a fresh clone: install uv, sync Python deps,
# install Nsight Systems CLI (for assignment2-systems profiling), and download
# datasets per assignment1-basics/README.md.
# Run from inside the cloned repo, or pass the repo root as a positional arg.
#
# Flags:
#   --skip-data   Skip the dataset download step (env: SKIP_DATA=1)
#   --skip-nsys   Skip Nsight Systems CLI install (env: SKIP_NSYS=1)
#   --nsys-only   Install Nsight Systems CLI only; skip uv sync, data, etc. (env: NSYS_ONLY=1)
#   -h, --help    Show this help and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

SKIP_DATA="${SKIP_DATA:-0}"
SKIP_NSYS="${SKIP_NSYS:-0}"
NSYS_ONLY="${NSYS_ONLY:-0}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-nsys) SKIP_NSYS=1; shift ;;
    --nsys-only) NSYS_ONLY=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    --)          shift; POSITIONAL+=("$@"); break ;;
    -*)          echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)           POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [[ "${NSYS_ONLY}" == "1" && "${SKIP_NSYS}" == "1" ]]; then
  echo "--nsys-only and --skip-nsys are mutually exclusive." >&2
  exit 2
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  echo "Installing uv (https://github.com/astral-sh/uv) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv was installed but is not on PATH. Add ~/.local/bin to PATH and re-run." >&2
    exit 1
  fi
}

fetch_file() {
  local url=$1
  local out=${2:-$(basename "$url")}
  if [[ -f "$out" ]]; then
    echo "  skip (exists): $out"
    return
  fi
  echo "  downloading: $out"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$url" -O "$out"
  else
    curl -fL --progress-bar -o "$out" "$url"
  fi
}

install_nsys() {
  if [[ "${SKIP_NSYS}" == "1" ]]; then
    echo "Skipping Nsight Systems CLI install (SKIP_NSYS=1)."
    return 0
  fi
  if command -v nsys >/dev/null 2>&1; then
    echo "nsys already installed: $(nsys --version 2>/dev/null | head -n1 || echo unknown)"
    return 0
  fi

  local os
  os="$(uname -s)"
  if [[ "${os}" != "Linux" ]]; then
    echo "Nsight Systems CLI auto-install only supported on Linux; skipping on ${os}."
    echo "  Get it from https://developer.nvidia.com/nsight-systems if needed."
    return 0
  fi
  if [[ ! -r /etc/os-release ]]; then
    echo "Cannot detect Linux distro (no /etc/os-release); skipping nsys install."
    return 0
  fi

  local ID="" ID_LIKE="" VERSION_ID=""
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID}" != "ubuntu" && "${ID}" != "debian" && "${ID_LIKE}" != *debian* ]]; then
    echo "Nsight Systems auto-install supports Ubuntu/Debian; detected ID=${ID:-unknown}."
    echo "  Install manually: https://developer.nvidia.com/nsight-systems"
    return 0
  fi

  local sudo_cmd=""
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd="sudo"
    else
      echo "Need root (or sudo) to install nsight-systems-cli; skipping."
      return 0
    fi
  fi

  local ver_id="${VERSION_ID//./}"
  local dpkg_arch cuda_arch
  dpkg_arch="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
  case "${dpkg_arch}" in
    amd64) cuda_arch="x86_64" ;;
    arm64) cuda_arch="sbsa" ;;
    *)     cuda_arch="${dpkg_arch}" ;;
  esac

  local repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${ID}${ver_id}/${cuda_arch}"
  echo "Installing Nsight Systems CLI from ${repo_url} ..."
  local tmp
  tmp="$(mktemp -d)"
  if ! curl -fLsS -o "${tmp}/cuda-keyring.deb" "${repo_url}/cuda-keyring_1.1-1_all.deb"; then
    echo "Failed to fetch cuda-keyring from ${repo_url}." >&2
    rm -rf "${tmp}"
    return 1
  fi
  ${sudo_cmd} dpkg -i "${tmp}/cuda-keyring.deb" || { rm -rf "${tmp}"; return 1; }
  rm -rf "${tmp}"

  # Some base images (e.g. RunPod's PyTorch images) ship a pre-existing CUDA
  # repo .list without a signed-by= keyring. After cuda-keyring drops its own
  # signed .list, `apt update` aborts with "Conflicting values set for option
  # Signed-By". Move any unsigned dup pointing at the same CUDA repo out of
  # sources.list.d entirely (renaming in-place leaves the file there and apt
  # warns about the unrecognized extension on every run).
  local backup_dir="/etc/apt/sources.list.d.disabled-by-cs336-setup"
  local list_file
  shopt -s nullglob
  for list_file in /etc/apt/sources.list.d/*.list; do
    if grep -q 'developer.download.nvidia.com/compute/cuda' "${list_file}" \
        && ! grep -q 'signed-by' "${list_file}"; then
      ${sudo_cmd} mkdir -p "${backup_dir}"
      echo "  disabling conflicting CUDA source: ${list_file} -> ${backup_dir}/"
      ${sudo_cmd} mv "${list_file}" "${backup_dir}/"
    fi
  done
  shopt -u nullglob

  ${sudo_cmd} apt-get update -y || return 1

  # Package name varies across CUDA repo versions:
  #   ubuntu22.04 ships a bare "nsight-systems-cli" metapackage,
  #   ubuntu24.04 only ships versioned "nsight-systems-cli-<ver>" packages.
  # Pick the newest available, preferring -cli variants.
  local nsys_pkg
  nsys_pkg="$(apt-cache pkgnames 2>/dev/null | grep -E '^nsight-systems-cli(-[0-9].*)?$' | sort -V | tail -1)"
  if [[ -z "${nsys_pkg}" ]]; then
    nsys_pkg="$(apt-cache pkgnames 2>/dev/null | grep -E '^nsight-systems(-[0-9].*)?$' | sort -V | tail -1)"
  fi
  if [[ -z "${nsys_pkg}" ]]; then
    echo "Could not find an nsight-systems package in the configured apt sources." >&2
    return 1
  fi
  echo "  installing apt package: ${nsys_pkg}"
  ${sudo_cmd} apt-get install -y "${nsys_pkg}" || return 1

  if ! command -v nsys >/dev/null 2>&1; then
    # Versioned packages drop nsys under /opt/nvidia/nsight-systems[-cli]/<ver>/bin
    # without putting it on PATH. Symlink the newest one into /usr/local/bin.
    local nsys_bin
    nsys_bin="$(ls -1 /opt/nvidia/nsight-systems-cli/*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1)"
    if [[ -n "${nsys_bin}" && -x "${nsys_bin}" ]]; then
      echo "  linking ${nsys_bin} -> /usr/local/bin/nsys"
      ${sudo_cmd} ln -sf "${nsys_bin}" /usr/local/bin/nsys
    fi
  fi

  if command -v nsys >/dev/null 2>&1; then
    echo "nsys installed: $(nsys --version 2>/dev/null | head -n1)"
  else
    echo "nsight-systems installed but nsys not on PATH; check /opt/nvidia/." >&2
  fi
}

if [[ "${NSYS_ONLY}" == "1" ]]; then
  echo "Installing Nsight Systems CLI only (--nsys-only); skipping uv sync, data download, and all other setup."
  if ! install_nsys; then
    echo "Nsight Systems install did not complete; install manually if needed." >&2
    exit 1
  fi
  echo "Done."
  exit 0
fi

# Resolve repo root: prefer directory containing this script, then cwd, then $1.
if [[ -f "${SCRIPT_DIR}/assignment1-basics/pyproject.toml" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${PWD}/assignment1-basics/pyproject.toml" ]]; then
  REPO_ROOT="${PWD}"
elif [[ -n "${1:-}" && -f "${1}/assignment1-basics/pyproject.toml" ]]; then
  REPO_ROOT="$(cd "$1" && pwd)"
else
  echo "Could not locate assignment1-basics/pyproject.toml." >&2
  echo "Run this script from inside the cloned repo, or pass the repo root as \$1." >&2
  exit 1
fi

ASSIGN="${REPO_ROOT}/assignment1-basics"
if [[ ! -f "${ASSIGN}/pyproject.toml" ]]; then
  echo "Expected ${ASSIGN}/pyproject.toml — is this the cs336-assignments repo?" >&2
  exit 1
fi

echo "Using repo root: ${REPO_ROOT}"
cd "${ASSIGN}"

ensure_uv
echo "Syncing Python environment (uv sync) ..."
uv sync

if ! install_nsys; then
  echo "Nsight Systems install did not complete; install manually if needed."
fi

if [[ "${SKIP_DATA}" == "1" ]]; then
  echo "Skipping data download (SKIP_DATA=1 / --skip-data)."
else
  echo "Downloading data (see README.md § Download data) ..."
  mkdir -p data
  (
    cd data
    fetch_file "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    fetch_file "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    fetch_file "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
    fetch_file "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
    if [[ -f owt_train.txt.gz && ! -f owt_train.txt ]]; then
      gunzip -f owt_train.txt.gz
    fi
    if [[ -f owt_valid.txt.gz && ! -f owt_valid.txt ]]; then
      gunzip -f owt_valid.txt.gz
    fi
  )
fi

# README uses assignment1-basics/data; byte_pair_encoding.py uses cs336_basics/data.
if [[ ! -e cs336_basics/data ]]; then
  echo "Linking cs336_basics/data -> ../data (for modules that expect data next to the package)"
  ln -sfn ../data cs336_basics/data
fi

echo
echo "Done. From ${ASSIGN} run tests with:"
echo "  uv run pytest"
echo
if ! command -v uv >/dev/null 2>&1; then
  echo "NOTE: uv was just installed but isn't in your current shell's PATH."
  echo "Run one of the following to pick it up, then retry:"
  echo '  source ~/.bashrc   # or: source ~/.zshrc / source ~/.profile'
  echo '  export PATH="$HOME/.local/bin:$PATH"'
fi
