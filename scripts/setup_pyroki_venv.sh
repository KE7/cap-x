#!/usr/bin/env bash
#
# Create the isolated GPU venv used by the pyroki IK/plan server.
#
# The pyroki server (capx/serving/launch_pyroki_server.py) runs GPU-enabled JAX
# (jax[cuda13], numpy 2.x), which is incompatible with the benchmark venvs that
# pin numpy 1.26.4. It therefore lives in its own dedicated venv, ``.venv-pyroki``
# (gitignored), and is reached over HTTP. ``capx/serving/launch_servers.py``
# launches the pyroki server with this venv's interpreter by default.
#
# Run this once from the repo root on any machine that needs to serve pyroki:
#
#     ./scripts/setup_pyroki_venv.sh
#
# Requires: uv (https://docs.astral.sh/uv/) on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv-pyroki"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found on PATH. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "==> Creating venv at ${VENV} (python 3.11)"
uv venv "${VENV}" --python 3.11

# Install from /tmp so uv does not resolve against the repo's pyproject/uv.lock
# (which pins numpy 1.26.4 for the benchmark stacks and would conflict with the
# GPU jax/numpy-2.x stack this server needs).
echo "==> Installing pyroki + GPU jax[cuda13] + server deps"
(
  cd /tmp
  uv pip install --python "${VENV}/bin/python" \
    "jax[cuda13]" \
    "pyroki@git+https://github.com/chungmin99/pyroki.git" \
    fastapi uvicorn pydantic scipy requests robot_descriptions yourdfpy tyro pyyaml
)

# Install capx itself (no deps — everything the server needs is installed above).
echo "==> Installing capx (editable, --no-deps)"
uv pip install -e "${REPO_ROOT}" --no-deps --python "${VENV}/bin/python"

echo "==> Done. Verify with:"
echo "    ${VENV}/bin/python -c \"import jax; print(jax.devices())\""
