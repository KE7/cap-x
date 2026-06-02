"""Tests for package-mode user-code execution in cap-x.

Covers both the new ``SimpleExecutor.run_package`` helper and the
``CodeExecutionEnvBase._exec_user_package`` / ``_exec`` dispatcher
introduced alongside the existing string-mode executor.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from capx.envs.tasks.base import (
    CodeExecutionEnvBase,
    SimpleExecutor,
)


# ---------------------------------------------------------------------------
# Helpers: minimal stand-ins for the low-level env and API surface so we can
# exercise the executor logic without booting a real simulator or API.
# ---------------------------------------------------------------------------


class _StubEnv:
    """Bare minimum low-level env surface that the executor touches."""

    def get_observation(self) -> dict[str, Any]:
        return {"stub_obs": True}


def _build_code_env() -> CodeExecutionEnvBase:
    """Instantiate CodeExecutionEnvBase without running its real __init__.

    The real __init__ builds sim envs and APIs; for unit tests we only need
    the handful of attributes the (string|package) executors consult.
    """
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = _StubEnv()  # type: ignore[assignment]
    env._apis = {}  # type: ignore[attr-defined]
    env._exec_globals = {  # type: ignore[attr-defined]
        "__name__": "__main__",
        "env": env.low_level_env,
        "APIS": env._apis,  # type: ignore[attr-defined]
        "INPUTS": {},
        "RESULT": None,
    }
    env._full_prompt = []  # type: ignore[attr-defined]
    env._task_prompt = None  # type: ignore[attr-defined]
    return env


def _write_pkg(root: Path, name: str, files: dict[str, str]) -> Path:
    """Write a Python package under ``root/name`` and return its path."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(files.get("__init__.py", ""))
    for rel, content in files.items():
        if rel == "__init__.py":
            continue
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return pkg


@pytest.fixture(autouse=True)
def _cleanup_sys_modules():
    """Snapshot/restore sys.modules so per-test packages don't leak."""
    before = set(sys.modules)
    yield
    for mod_name in list(sys.modules):
        if mod_name not in before:
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# 1. String-mode unchanged
# ---------------------------------------------------------------------------


def test_string_mode_unchanged() -> None:
    env = _build_code_env()
    out = env._exec_user_code("RESULT = 42")
    assert out["ok"] is True
    assert out["result"] == 42


# ---------------------------------------------------------------------------
# 2. Package-mode happy path
# ---------------------------------------------------------------------------


def test_package_mode_happy_path(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_happy",
        {
            "main.py": textwrap.dedent(
                """
                def run(ctx):
                    ctx["RESULT"] = 123
                    return 123
                """
            ),
        },
    )
    env = _build_code_env()
    out = env._exec_user_package(pkg)
    assert out["ok"] is True
    assert out["result"] == 123


# ---------------------------------------------------------------------------
# 3. Package-mode with a submodule import
# ---------------------------------------------------------------------------


def test_package_mode_submodule_import(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_sub",
        {
            "helpers.py": "def make() -> int:\n    return 7\n",
            "main.py": textwrap.dedent(
                """
                from pkg_sub.helpers import make

                def run(ctx):
                    return make() * 3
                """
            ),
        },
    )
    env = _build_code_env()
    out = env._exec_user_package(pkg)
    assert out["ok"] is True
    assert out["result"] == 21


# ---------------------------------------------------------------------------
# 4. Fresh-import: mutating the package between calls is visible
# ---------------------------------------------------------------------------


def test_package_mode_fresh_import_on_mutation(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_mut",
        {
            "helpers.py": "VALUE = 1\n",
            "main.py": textwrap.dedent(
                """
                from pkg_mut.helpers import VALUE

                def run(ctx):
                    return VALUE
                """
            ),
        },
    )
    env = _build_code_env()
    first = env._exec_user_package(pkg)
    assert first["ok"] is True
    assert first["result"] == 1

    # Mutate the helper module's source between calls.
    (pkg / "helpers.py").write_text("VALUE = 999\n")

    second = env._exec_user_package(pkg)
    assert second["ok"] is True
    assert second["result"] == 999


# ---------------------------------------------------------------------------
# 5. Package-mode error path
# ---------------------------------------------------------------------------


def test_package_mode_error_propagates(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_err",
        {
            "main.py": textwrap.dedent(
                """
                def run(ctx):
                    raise RuntimeError("boom from user package")
                """
            ),
        },
    )
    env = _build_code_env()
    out = env._exec_user_package(pkg)
    assert out["ok"] is False
    assert "boom from user package" in out["stderr"]
    # Traceback should include the original exception type.
    assert "RuntimeError" in out["stderr"]


# ---------------------------------------------------------------------------
# 6. Package-mode: missing `run` entry point
# ---------------------------------------------------------------------------


def test_package_mode_missing_run(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_no_run",
        {
            "main.py": "# intentionally no run() defined\nVALUE = 1\n",
        },
    )
    env = _build_code_env()
    out = env._exec_user_package(pkg)
    assert out["ok"] is False
    assert "must define `run(ctx)`" in out["stderr"]


# ---------------------------------------------------------------------------
# 7. Dispatcher branches correctly on Path vs str
# ---------------------------------------------------------------------------


def test_dispatcher_branches(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_disp",
        {
            "main.py": textwrap.dedent(
                """
                def run(ctx):
                    return "from-package"
                """
            ),
        },
    )
    env = _build_code_env()

    # Path input -> package mode.
    pkg_out = env._exec(pkg)
    assert pkg_out["ok"] is True
    assert pkg_out["result"] == "from-package"

    # String input that happens to be a directory -> package mode too.
    pkg_out_str = env._exec(str(pkg))
    assert pkg_out_str["ok"] is True
    assert pkg_out_str["result"] == "from-package"

    # Plain code string -> string mode, unchanged behavior.
    code_out = env._exec("RESULT = 1")
    assert code_out["ok"] is True
    assert code_out["result"] == 1


# ---------------------------------------------------------------------------
# SimpleExecutor parity check: run_package returns the same contract as run
# ---------------------------------------------------------------------------


def test_simple_executor_run_package(tmp_path: Path) -> None:
    pkg = _write_pkg(
        tmp_path,
        "pkg_simple",
        {
            "main.py": textwrap.dedent(
                """
                def run(ctx):
                    ctx["RESULT"] = ctx["INPUTS"].get("x", 0) + 5
                    return ctx["RESULT"]
                """
            ),
        },
    )
    exec_ = SimpleExecutor(env=_StubEnv(), apis={})  # type: ignore[arg-type]
    out = exec_.run_package(pkg, inputs={"x": 10})
    assert out["ok"] is True
    assert out["result"] == 15

    # Missing entry point surfaces as ok=False with an error field.
    _write_pkg(
        tmp_path,
        "pkg_simple_broken",
        {"main.py": "# no run here\n"},
    )
    bad = exec_.run_package(tmp_path / "pkg_simple_broken")
    assert bad["ok"] is False
    assert "error" in bad
