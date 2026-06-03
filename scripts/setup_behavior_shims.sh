#!/usr/bin/env bash
# Create + export the BEHAVIOR / R1Pro PYTHONPATH shims (aarch64 + source-built Isaac).
#
# WHY: The source-built aarch64 Isaac Sim 5.1 puts its OWN bundled copies of
# `websockets` and `typing_extensions` on PYTHONPATH via setup_python_env.sh.
# Isaac's bundled `typing_extensions` is stale and lacks `Sentinel`, so
# omnigibson / websockets imports break. We surgically prepend ONLY the b1k venv's
# `websockets` package and `typing_extensions.py` so those two win over Isaac's
# bundled copies — WITHOUT shadowing the rest of Isaac's bundled deps (which is what
# prepending the whole venv site-packages would do).
#
# This replaces the old hand-made /tmp/ws_shim + /tmp/te_shim symlinks with a
# committed, reproducible step.
#
# USAGE — source it AFTER `source "$ISAAC_PATH/setup_python_env.sh"` (see
# docs/spark-aarch64-setup.md §10.4):
#     source scripts/setup_behavior_shims.sh
# It creates the shim dirs, exports WS_SHIM + TE_SHIM, and prepends them to
# PYTHONPATH. Override defaults via env vars:
#     B1K_VENV   b1k venv root      (default: capx/third_party/b1k/.venv)
#     SHIM_DIR   where to materialize the shim dirs (default: ${TMPDIR:-/tmp}/capx-behavior-shims)

set -u

B1K_VENV="${B1K_VENV:-capx/third_party/b1k/.venv}"
SHIM_DIR="${SHIM_DIR:-${TMPDIR:-/tmp}/capx-behavior-shims}"

_fail() { echo "setup_behavior_shims: ERROR: $*" >&2; return 1 2>/dev/null || exit 1; }

venv_py="$B1K_VENV/bin/python"
[[ -x "$venv_py" ]] || _fail "b1k venv python not found at '$venv_py' (set B1K_VENV)"

# Ask the venv where its purelib site-packages is (works for py3.10 or py3.11).
site_pkgs="$("$venv_py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
  || _fail "could not determine site-packages from $venv_py"

ws_src="$site_pkgs/websockets"
te_src="$site_pkgs/typing_extensions.py"
[[ -e "$ws_src" ]] || _fail "missing '$ws_src' in the b1k venv (pip install websockets)"
[[ -e "$te_src" ]] || _fail "missing '$te_src' in the b1k venv (pip install typing_extensions)"

WS_SHIM="$SHIM_DIR/ws_shim"
TE_SHIM="$SHIM_DIR/te_shim"
mkdir -p "$WS_SHIM" "$TE_SHIM"
ln -sfn "$ws_src" "$WS_SHIM/websockets"
ln -sfn "$te_src" "$TE_SHIM/typing_extensions.py"

export WS_SHIM TE_SHIM
# Shims FIRST so the venv copies win over Isaac's bundled ones.
export PYTHONPATH="$WS_SHIM:$TE_SHIM:${PYTHONPATH:-}"

echo "setup_behavior_shims: WS_SHIM=$WS_SHIM TE_SHIM=$TE_SHIM (prepended to PYTHONPATH)"
