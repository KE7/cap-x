#!/usr/bin/env bash
#
# Install cuRobo into a target venv, PREFERRING a prebuilt wheel published as a
# GitHub release on the KE7/curobo fork, and FALLING BACK to a from-source build
# of the fork (capx/third_party/curobo) when no matching prebuilt wheel is
# available. Mirrors the spark-vllm-docker "prebuilt-wheel-then-build" pattern.
#
# Why: the cuRobo CUDA extensions take ~20-40 min of nvcc to compile on the
# DGX Spark (GB10). The prebuilt wheel skips that for the platform it was built
# for (aarch64 / cp311 / CUDA 13 / sm_121, torch 2.12.0+cu130). On any other
# platform (or if the download/import fails) we transparently build from source.
#
# The fork's `aarch64/cuda13-lerp-fix` branch already carries the C++20
# std::lerp guard committed in helper_math.h, so NO working-tree patch is needed
# for the source build (this is why patches/curobo-lerp.patch can be dropped).
#
# Usage:
#   scripts/install_curobo.sh <venv_dir>
#
# Honors (inherited from the calling setup script):
#   CUDA_HOME, CUDA_PATH, TORCH_CUDA_ARCH_LIST  -- required for the source build
#   UV_LOCK_TIMEOUT                             -- raised for the editable build
# Overrides:
#   CUROBO_WHEEL_URL=<url>     -- use a different prebuilt wheel
#   CUROBO_FORCE_SOURCE=1      -- skip the prebuilt wheel; always build from source
set -euo pipefail

VENV="${1:?usage: install_curobo.sh <venv_dir>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUROBO_DIR="${REPO_ROOT}/capx/third_party/curobo"
PY="${VENV}/bin/python"

# Prebuilt wheel published from KE7/curobo (built from branch
# aarch64/cuda13-lerp-fix @ 30eafef; cp311 / linux_aarch64 / CUDA 13 / sm_121,
# torch 2.12.0+cu130). See docs/spark-aarch64-setup.md §curobo for provenance.
CUROBO_WHEEL_URL="${CUROBO_WHEEL_URL:-https://github.com/KE7/curobo/releases/download/v-cu13-aarch64-30eafef/nvidia_curobo-0.7.8.post1.dev1-cp311-cp311-linux_aarch64.whl}"

build_from_source() {
  echo "==> curobo: building from source (${CUROBO_DIR})"
  # No patch: the fork already guards scalar lerp against C++20 std::lerp.
  UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-3600}" \
  uv pip install --python "${PY}" \
    --no-build-isolation --reinstall-package nvidia-curobo \
    -e "${CUROBO_DIR}"
}

install_prebuilt() {
  if [ "${CUROBO_FORCE_SOURCE:-0}" = "1" ]; then
    echo "==> curobo: CUROBO_FORCE_SOURCE=1 set; skipping prebuilt wheel."
    return 1
  fi
  # The prebuilt wheel is platform-specific; only use it where it applies.
  local arch py_tag
  arch="$("${PY}" -c 'import platform; print(platform.machine())')"
  py_tag="$("${PY}" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
  if [ "${arch}" != "aarch64" ] || [ "${py_tag}" != "cp311" ]; then
    echo "==> curobo: prebuilt wheel is aarch64/cp311 only (this env: ${arch}/${py_tag}); will build from source."
    return 1
  fi

  local tmp whl
  tmp="$(mktemp -d)"
  whl="${tmp}/$(basename "${CUROBO_WHEEL_URL}")"
  echo "==> curobo: fetching prebuilt wheel: ${CUROBO_WHEEL_URL}"
  if ! curl -fSL --retry 3 -o "${whl}" "${CUROBO_WHEEL_URL}"; then
    echo "==> curobo: prebuilt wheel download failed; will build from source."
    rm -rf "${tmp}"; return 1
  fi
  if ! uv pip install --python "${PY}" \
        --index-strategy unsafe-best-match --extra-index-url https://pypi.org/simple \
        --reinstall-package nvidia-curobo "${whl}"; then
    echo "==> curobo: prebuilt wheel install failed; will build from source."
    rm -rf "${tmp}"; return 1
  fi
  rm -rf "${tmp}"
  # Sanity: the compiled CUDA extension must actually import in this env.
  # torch must be imported first so its libc10.so / libtorch are on the loader
  # path (the curobolib .so links them); otherwise the import is a false negative.
  if ! "${PY}" -c 'import torch; import curobo.curobolib.kinematics_fused_cu' 2>/dev/null; then
    echo "==> curobo: prebuilt CUDA extension failed to import here; will build from source."
    return 1
  fi
  echo "==> curobo: installed prebuilt wheel OK (skipped the nvcc build)."
  return 0
}

if ! install_prebuilt; then
  build_from_source
fi

echo "==> curobo: verifying import"
"${PY}" -c 'import curobo; print("curobo", curobo.__version__, "OK")'
