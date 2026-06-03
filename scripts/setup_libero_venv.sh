#!/usr/bin/env bash
#
# Reproducible setup of the LIBERO benchmark venv (.venv-libero) on aarch64 /
# DGX Spark (NVIDIA GB10, Blackwell sm_121, CUDA 13).
#
# LIBERO needs robosuite 1.4 and conflicts with the robosuite-1.5 venv (.venv),
# so it lives in its own venv. This installs the [libero] + [contactgraspnet]
# extras. See setup_robosuite_venv.sh for the robosuite venv and
# docs/spark-aarch64-setup.md for the full machine context.
#
# Why a dedicated script (and not `uv sync`): on aarch64 the project's
# `[tool.uv] environments` is pinned to x86_64 only, so `uv sync` hard-errors.
# Default PyPI torch on aarch64 is also CPU-only. This uses the proven
# `uv pip install` path with the cu130 torch recipe + the aarch64 overrides.
#
# Run once from the repo root:
#     ./scripts/setup_libero_venv.sh
#
# Requires: uv on PATH; system CUDA 13 at /usr/local/cuda (nvcc); git submodules
# initialised (`git submodule update --init --recursive`, incl. verl which
# LIBERO metadata generation needs).
#
# Idempotent: re-running recreates .venv-libero and reinstalls. Safe to re-run.
#
# Verified path: rebuilt at Python 3.11.14; 206 pkgs resolved; open3d 0.18.0 from
# a real aarch64 cp311 wheel; cu130 torch preserved; curobo editable build OK.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv-libero"
OVERRIDES="${REPO_ROOT}/scripts/overrides/aarch64-overrides.txt"
PYVER="3.11"   # 3.11 (NOT 3.12): open3d==0.18.0 has no cp312 aarch64 wheel.

# --- The shared cu130 / Blackwell build env (must be exported before any build) ---
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1+PTX}"   # GB10 = sm_121

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found on PATH. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc (need CUDA 13 toolkit)." >&2
  exit 1
fi
if [ ! -f "${OVERRIDES}" ]; then
  echo "error: overrides file missing: ${OVERRIDES}" >&2
  exit 1
fi

echo "==> [1/4] Creating venv at ${VENV} (python ${PYVER})"
# --clear wipes any existing venv so reruns are a clean recreate (uv pip install
# is additive, so without this stale packages from a prior run could survive).
uv venv --clear "${VENV}" --python "${PYVER}"

echo "==> [2/4] Installing cu130 CUDA torch (aarch64 GB10)"
uv pip install --python "${VENV}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  torch==2.12.0+cu130 torchvision==0.27.0+cu130 torchaudio==2.11.0+cu130

echo "==> [3/4] Installing capx [libero,contactgraspnet] with aarch64 overrides"
# --overrides applies open3d==0.18.0 + decord gate (+ mirrored pyproject overrides).
# --no-build-isolation-package nvidia-curobo so curobo builds against cu130 torch.
# UV_LOCK_TIMEOUT raised because the editable curobo build can hold the cache lock.
UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-3600}" \
uv pip install --python "${VENV}/bin/python" \
  --no-build-isolation-package nvidia-curobo \
  --overrides "${OVERRIDES}" \
  -e "${REPO_ROOT}[libero,contactgraspnet]"

echo "==> [4/4] Installing curobo (prefer prebuilt wheel, fall back to source build)"
# Prefers the prebuilt CUDA-13/aarch64/sm_121 wheel from the KE7/curobo release
# (skips the ~20-40 min nvcc build); falls back to a from-source build of the
# fork. The fork carries the C++20 std::lerp guard committed, so no patch needed.
"${REPO_ROOT}/scripts/install_curobo.sh" "${VENV}"

echo "==> Verifying imports"
"${VENV}/bin/python" - <<'PY'
import sys
import torch, libero, robosuite, contact_graspnet_pytorch, curobo, open3d
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-CUDA")
print("open3d", open3d.__version__)
print("robosuite", robosuite.__version__)
print("libero OK; contact_graspnet_pytorch OK; curobo OK")
# Assert the architecture-critical invariants this setup documents; fail loudly
# (non-zero exit) on any violation rather than silently printing a broken state.
errs = []
if not torch.cuda.is_available():
    errs.append("torch is CPU-only (expected cu130 GPU build on GB10)")
if open3d.__version__ != "0.18.0":
    errs.append(f"open3d {open3d.__version__} != 0.18.0 (aarch64 cp311 pin)")
try:
    import decord  # noqa: F401
    errs.append("decord is importable but must be gated out (sam3-train-only)")
except ModuleNotFoundError:
    print("decord absent (expected, gated -- sam3-train-only)")
if errs:
    sys.exit("FAIL: " + "; ".join(errs))
print("LIBERO VENV OK")
PY

echo "==> Done. Expected: torch 2.12.0+cu130, cuda True, NVIDIA GB10, open3d 0.18.0, robosuite 1.4.0."
