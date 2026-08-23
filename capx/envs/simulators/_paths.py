"""Path resolution helpers for simulator config loading.

Robosuite's controller loader opens paths via ``os.getcwd()``.  Callers
inside cap-x must pass absolute paths to it, independent of the caller's
working directory.
"""
from __future__ import annotations

from pathlib import Path

# capx/envs/simulators/_paths.py  ->  parents[3] == cap-x repo root
_CAPX_ROOT: Path = Path(__file__).resolve().parents[3]


def resolve_relative_to_capx(path: str | Path) -> str:
    """Return ``path`` as an absolute str, anchored at the cap-x repo root
    when the input is relative."""
    p = Path(path)
    return str(p if p.is_absolute() else _CAPX_ROOT / p)
