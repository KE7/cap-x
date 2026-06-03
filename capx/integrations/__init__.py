"""capx.integrations package.

The eager API registrations live in :mod:`capx.integrations._register_apis` and
are imported here so that importing ``capx.integrations`` (directly or via any
submodule) populates the API registry as a side effect — exactly as before.

EXCEPTION: when the environment variable ``CAPX_PYROKI_SERVER_ONLY=1`` is set,
the heavy registrations (which pull in robosuite / torch / open3d via the franka
and libero APIs) are skipped. This is used by the isolated pyroki GPU service
venv (``.venv-pyroki``), which runs the IK/plan HTTP server and only needs
``capx.integrations.motion.pyroki_snippets``. Benchmark venvs never set this
flag, so their behavior is unchanged.
"""

import os as _os

if _os.environ.get("CAPX_PYROKI_SERVER_ONLY") != "1":
    from . import _register_apis  # noqa: F401  -- triggers API registrations
