"""Trial orchestration for CaP-X environments.

Handles batch execution, parallel worker dispatch, retry logic, and
wall-clock timeouts for running code-generation trials.  The actual
single-trial execution lives in :mod:`capx.envs.trial`.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _run_single_trial,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _print_and_save_summary,
    _save_trial_artifacts,
    run_server_proc,
)
from capx.utils.parallel_eval import run_parallel_with_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-trial wall-clock cap (SIGALRM). This is the value the benchmark commits on
# origin/main. A local edit had raised it to 10800 for a separate 122B Isaac run
# (see m3-gpu-rerun); that edit was removed so LIBERO evaluation is faithful to
# the benchmark. Preserved at /home/batman/.cache/runner.py.with-10800 if the
# Isaac run needs it back -- do NOT restore it while LIBERO is running.
TRIAL_TIMEOUT_SECONDS = 1000
MAX_TRIAL_RETRIES = 3


# ---------------------------------------------------------------------------
# API server helpers
# ---------------------------------------------------------------------------

def _start_api_servers(
    api_servers: list | None, wait_timeout: float = 120.0
) -> list:
    """Launch any API server sub-processes defined in the config.

    Skips servers whose port is already in use (e.g. started externally).
    After launching, waits until all servers are accepting connections.
    """
    import socket
    import time

    procs = []
    ports_to_wait: list[tuple[str, int]] = []
    if api_servers is not None:
        for api_server in api_servers:
            port = api_server.get("port")
            host = api_server.get("host", "127.0.0.1")
            if port is not None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, int(port))) == 0:
                        print(f"API server on {host}:{port} already running, skipping")
                        continue
            proc = run_server_proc(api_server)
            procs.append(proc)
            print(f"API server {api_server} started")
            if port is not None:
                ports_to_wait.append((host, int(port)))

    # Wait for all launched servers to accept connections
    if ports_to_wait:
        deadline = time.time() + wait_timeout
        for host, port in ports_to_wait:
            while time.time() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, port)) == 0:
                        print(f"API server on {host}:{port} is ready")
                        break
                time.sleep(1.0)
            else:
                print(f"WARNING: API server on {host}:{port} not ready after {wait_timeout}s")

    return procs


def _stop_api_servers(server_procs: list) -> None:
    """Terminate API server sub-processes, escalating until they are actually gone.

    A service that outlives this call is not merely leaked memory and a held
    port.  It also still owns the stdout/stderr it inherited from the
    evaluator, so a harness that captures evaluator output with
    ``subprocess.run(..., capture_output=True)`` never observes EOF on those
    pipes and blocks *forever* -- the evaluator process itself has already
    exited and been reaped into a zombie, so there is nothing left to time out
    against.  Observed on sparky: one PyRoki left on 8216 hung `helix evolve`
    indefinitely at 0% CPU after the evaluation had otherwise finished.

    A plain ``terminate()`` plus a short grace is not sufficient.  These
    services import JAX/torch and load models at startup, and a process that
    is still inside that import cannot service SIGTERM promptly; the previous
    5s grace silently gave up and abandoned it.  Escalate to SIGKILL and
    verify, because "probably dead" is what caused the hang.
    """
    for proc in server_procs:
        if not proc.is_alive():
            proc.join(timeout=1.0)
            continue
        proc.terminate()
        proc.join(timeout=15.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        if proc.is_alive():
            print(
                f"WARNING: api server pid {getattr(proc, 'pid', '?')} survived "
                f"SIGKILL; it still holds its port and the inherited stdout "
                f"pipe, which can hang a capturing parent.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Evaluation service layer
# ---------------------------------------------------------------------------
#
# Infrastructure for harnesses that fan a single evaluation out over several
# rollout worker processes.  Some CaP-X services (segmentation, grasp
# proposal) are stateless and expensive to load, so one instance is shared by
# the whole evaluation; others (motion planning) hold per-episode state and
# must be private to a worker.
#
# Nothing in this layer reads a module-level timeout or retry constant: every
# caller passes its own numbers in, so one experiment's constants can never
# silently become another's.

#: Default scope for each known CaP-X service entry point.
#:
#: ``"session"`` -> one instance shared by the whole evaluation.
#: ``"worker"``  -> one instance per parallel rollout worker.
EVALUATION_SERVICE_SCOPE: dict[str, str] = {
    "capx.serving.launch_sam3_server.main": "session",
    "capx.serving.launch_sam2_server.main": "session",
    "capx.serving.launch_contact_graspnet_server.main": "session",
    "capx.serving.launch_owlvit_server.main": "session",
    "capx.serving.launch_pyroki_server.main": "worker",
}

#: Environment variable a worker-scoped service's client reads to find it.
EVALUATION_SERVICE_URL_ENV: dict[str, str] = {
    "capx.serving.launch_pyroki_server.main": "PYROKI_SERVICE_URL",
}

#: Routes a session-scoped service must expose for its identity to be confirmed
#: before the evaluation adopts an already-running instance of it.  Port numbers
#: are not proof of identity: PyRoki also serves "/plan", so adopting whatever
#: answers on 8215 could hand the evaluation a motion planner instead of a grasp
#: planner.  Probed via the FastAPI "/openapi.json" document.
EVALUATION_SERVICE_REQUIRED_ROUTES: dict[str, tuple[str, ...]] = {
    "capx.serving.launch_sam3_server.main": ("/segment", "/segment_point"),
    "capx.serving.launch_sam2_server.main": ("/segment", "/segment_point"),
    "capx.serving.launch_contact_graspnet_server.main": ("/plan", "/plan_point_clouds"),
    "capx.serving.launch_owlvit_server.main": ("/detect",),
}

#: Request-schema properties that identify a service when its ROUTES do not.
#:
#: SAM2 and SAM3 expose exactly the same two paths ("/segment",
#: "/segment_point") and set no FastAPI title, so a route check alone would
#: adopt a resident SAM2 as SAM3 and report "identity verified" -- silently
#: swapping the segmentation model under every scored trial.  Their request
#: bodies differ, and that difference is visible in the same /openapi.json:
#: SAM3's SegmentRequest carries ``text_prompt`` (it is text-prompted), SAM2's
#: carries ``box``/``max_masks``.  Checked as {schema_name: (property, ...)}.
EVALUATION_SERVICE_REQUIRED_SCHEMA: dict[str, dict[str, tuple[str, ...]]] = {
    "capx.serving.launch_sam3_server.main": {"SegmentRequest": ("text_prompt",)},
    "capx.serving.launch_sam2_server.main": {"SegmentRequest": ("box",)},
}

#: Ports an evaluation service may never be allocated, and who owns them.
RESERVED_EVALUATION_PORTS: dict[int, str] = {
    8122: "Molmo vLLM (hard-coded in capx/integrations/vision/molmo.py)",
    8129: "LIBERO development sidecar",
}

#: Ceiling on parallel evaluation workers; bounds the worker port span.
MAX_EVALUATION_WORKER_SLOTS = 20

#: Keys this layer consumes that must not be forwarded to ``instantiate``.
_EVALUATION_SERVICE_META_KEYS = ("scope", "url_env", "required_routes")


def evaluation_api_server_scope(api_server: dict) -> str:
    """Return ``"session"`` or ``"worker"`` for one ``api_servers`` entry.

    An explicit ``scope`` key wins; :data:`EVALUATION_SERVICE_SCOPE` is the
    default.  Raises for an unclassifiable service rather than defaulting,
    because a wrong default silently un-shares the vision services or silently
    shares a per-worker service -- neither of which shows up in a diff.
    """
    explicit = api_server.get("scope")
    if explicit is not None:
        if explicit not in ("session", "worker"):
            raise ValueError(
                f"api_server scope must be 'session' or 'worker', got {explicit!r} "
                f"for {api_server.get('_target_')!r}"
            )
        return str(explicit)
    target = api_server.get("_target_")
    try:
        return EVALUATION_SERVICE_SCOPE[target]
    except (KeyError, TypeError):
        raise ValueError(
            f"No evaluation scope known for api_server {target!r}. Add it to "
            f"EVALUATION_SERVICE_SCOPE in capx/envs/runner.py, or set an explicit "
            f"'scope: session|worker' on that entry. Refusing to guess: a wrong "
            f"scope silently changes whether the service is shared across workers."
        ) from None


def _evaluation_service_label(api_server: dict) -> str:
    return str(api_server.get("_target_", "<unnamed api_server>"))


def _evaluation_service_key(api_server: dict) -> str:
    """Stable identity for de-duplicating identical service declarations."""
    return json.dumps(api_server, sort_keys=True, default=repr)


def _evaluation_service_launch_config(api_server: dict) -> dict:
    """Copy of ``api_server`` with this layer's own keys removed."""
    cfg = copy.deepcopy(dict(api_server))
    for key in _EVALUATION_SERVICE_META_KEYS:
        cfg.pop(key, None)
    return cfg


def _evaluation_service_url_env(api_server: dict) -> str:
    """Name of the env var a client reads to reach this worker-scoped service."""
    explicit = api_server.get("url_env")
    if explicit:
        return str(explicit)
    target = api_server.get("_target_")
    try:
        return EVALUATION_SERVICE_URL_ENV[target]
    except (KeyError, TypeError):
        raise ValueError(
            f"No client URL environment variable known for worker-scoped api_server "
            f"{target!r}. Add it to EVALUATION_SERVICE_URL_ENV in capx/envs/runner.py, "
            f"or set an explicit 'url_env: <NAME>' on that entry."
        ) from None


def _evaluation_service_required_routes(api_server: dict) -> tuple[str, ...]:
    """Routes that prove a running server really is this service."""
    explicit = api_server.get("required_routes")
    if explicit:
        return tuple(str(route) for route in explicit)
    target = api_server.get("_target_")
    try:
        return EVALUATION_SERVICE_REQUIRED_ROUTES[target]
    except (KeyError, TypeError):
        raise ValueError(
            f"No identity routes known for session-scoped api_server {target!r}, so a "
            f"server already listening on its port cannot be verified before adoption. "
            f"Add it to EVALUATION_SERVICE_REQUIRED_ROUTES in capx/envs/runner.py, or "
            f"set an explicit 'required_routes: [...]' on that entry. Refusing to adopt "
            f"an unidentifiable server."
        ) from None


def _fetch_openapi_document(host: str, port: int, timeout: float) -> dict | None:
    """The server's OpenAPI document, or None if unreadable."""
    url = f"http://{host}:{int(port)}/openapi.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            document = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return document if isinstance(document, dict) else None


def _fetch_openapi_paths(host: str, port: int, timeout: float) -> set[str] | None:
    """Route set advertised by the server on ``host:port``, or None if unreadable."""
    document = _fetch_openapi_document(host, port, timeout)
    if document is None:
        return None
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return None
    return set(paths)


def _openapi_schema_properties(document: dict, schema_name: str) -> set[str]:
    """Property names of one component schema in an OpenAPI document."""
    schemas = (document.get("components") or {}).get("schemas") or {}
    schema = schemas.get(schema_name) or {}
    props = schema.get("properties") or {}
    return set(props) if isinstance(props, dict) else set()


def _verify_resident_evaluation_service(
    api_server: dict, host: str, port: int, *, probe_timeout: float
) -> int | None:
    """Confirm the server already on ``host:port`` is ``api_server``; else raise.

    Returns the owning PID when it can be determined.  Raises rather than
    adopting anything it cannot positively identify: silently adopting the
    wrong server is the failure mode this whole check exists to prevent.
    """
    role = _evaluation_service_label(api_server)
    required = _evaluation_service_required_routes(api_server)
    document = _fetch_openapi_document(host, port, probe_timeout)
    served = None
    if document is not None:
        paths = document.get("paths")
        served = set(paths) if isinstance(paths, dict) else None
    pid = _evaluation_port_owner_pid(port)
    owner = f"pid {pid}" if pid is not None else "an unidentified process"
    if served is None:
        raise RuntimeError(
            f"Refusing to adopt the server already listening on {host}:{port} as "
            f"{role}: it returned no readable /openapi.json within {probe_timeout}s, "
            f"so its identity cannot be confirmed. Expected routes "
            f"{list(required)}. That port is held by {owner}. Stop it, or start the "
            f"correct service there."
        )
    missing = sorted(set(required) - served)
    if missing:
        raise RuntimeError(
            f"Refusing to adopt the server already listening on {host}:{port} as "
            f"{role}: it is a different service. Expected routes {list(required)}, "
            f"missing {missing}; it serves {sorted(served)}. That port is held by "
            f"{owner}. Stop it, or start the correct service there."
        )

    # Routes matched. For services whose routes are NOT unique (SAM2 vs SAM3),
    # discriminate on the request schema before adopting.
    for schema_name, props in (
        EVALUATION_SERVICE_REQUIRED_SCHEMA.get(str(api_server.get("_target_")), {})
    ).items():
        found = _openapi_schema_properties(document, schema_name)
        missing_props = sorted(set(props) - found)
        if missing_props:
            raise RuntimeError(
                f"Refusing to adopt the server already listening on {host}:{port} as "
                f"{role}: its routes match but its request schema does not, so it is "
                f"a different model behind the same API. {schema_name} is missing "
                f"{missing_props}; it declares {sorted(found)}. (SAM2 and SAM3 serve "
                f"identical paths -- this check is what separates them.) That port is "
                f"held by {owner}. Stop it, or start the correct service there."
            )
    return pid


def _deduplicated_evaluation_api_servers(
    api_server_sets: Iterable[list | None], scope: str
) -> list[dict]:
    """Distinct ``scope``-scoped service declarations across several configs."""
    seen: set[str] = set()
    ordered: list[dict] = []
    by_port: dict[int, dict] = {}
    for api_servers in api_server_sets or []:
        if not api_servers:
            continue
        for api_server in api_servers:
            entry = dict(api_server)
            if evaluation_api_server_scope(entry) != scope:
                continue
            key = _evaluation_service_key(entry)
            if key in seen:
                continue
            port = entry.get("port")
            if port is not None:
                clash = by_port.get(int(port))
                if clash is not None:
                    raise ValueError(
                        f"Two different {scope}-scoped api_server declarations both "
                        f"claim port {int(port)}: {_evaluation_service_label(clash)!r} "
                        f"and {_evaluation_service_label(entry)!r}. Give them distinct "
                        f"ports; refusing to guess which one wins."
                    )
                by_port[int(port)] = entry
            seen.add(key)
            ordered.append(entry)
    return ordered


def _evaluation_port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, int(port))) == 0


def _evaluation_port_owner_pid(port: int) -> int | None:
    """Best-effort PID of whatever is listening on ``port`` (None if unknown)."""
    try:
        completed = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, text=True, timeout=5.0
        )
    except Exception:
        return None
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        if fields[3].rsplit(":", 1)[-1] != str(int(port)):
            continue
        match = re.search(r"pid=(\d+)", line)
        if match is not None:
            return int(match.group(1))
    return None


def _claim_evaluation_port(
    host: str, port: int, *, role: str, allow_existing: bool
) -> bool:
    """Validate ``port`` for ``role``; return True when we should start a server.

    Returns False only when the port is already served and ``allow_existing``
    permits adopting it.  Unlike :func:`_start_api_servers`, an occupied port is
    an error by default: an evaluation must never silently adopt a server whose
    configuration and GPU placement it does not know.
    """
    reserved = RESERVED_EVALUATION_PORTS.get(int(port))
    if reserved is not None:
        raise RuntimeError(
            f"Evaluation service {role} would bind reserved port {port} ({reserved}). "
            f"Change the declared port or reduce the worker-slot span."
        )
    if not _evaluation_port_is_open(host, int(port)):
        return True
    if allow_existing:
        return False
    pid = _evaluation_port_owner_pid(int(port))
    owner = f", owned by pid {pid}" if pid is not None else ""
    raise RuntimeError(
        f"Port {host}:{port} is already in use{owner}; refusing to adopt it for "
        f"{role}. Stop that listener, or pass allow_existing=True if reusing an "
        f"already-running service is intended."
    )


def _wait_for_evaluation_services(
    pending: list[tuple[str, int, str, Any]], wait_timeout: float
) -> None:
    """Block until every launched service accepts connections, else raise."""
    if not pending:
        return
    deadline = time.monotonic() + float(wait_timeout)
    for host, port, role, proc in pending:
        while True:
            if _evaluation_port_is_open(host, port):
                print(f"Evaluation service {role} on {host}:{port} is ready")
                break
            if proc is not None and not proc.is_alive():
                raise RuntimeError(
                    f"Evaluation service {role} exited with code {proc.exitcode} "
                    f"before accepting connections on {host}:{port}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Evaluation service {role} did not accept connections on "
                    f"{host}:{port} within {wait_timeout}s"
                )
            time.sleep(0.5)


def _run_server_proc_with_env(api_cfg, child_env: dict[str, str]) -> None:
    """Service subprocess entry point that applies ``child_env`` first.

    ``run_server_proc`` starts a ``multiprocessing.Process``, which inherits the
    parent's ``os.environ`` at fork/spawn time.  Setting e.g.
    ``CUDA_VISIBLE_DEVICES`` in the parent to place a service would therefore
    leak into every rollout worker started afterwards and silently colocate the
    vision services on the rollout device.  Applying it here -- inside the
    child, before ``instantiate`` imports/initialises any CUDA or JAX backend --
    keeps the override scoped to this one service.
    """
    if child_env:
        os.environ.update({str(k): str(v) for k, v in child_env.items()})
    instantiate(api_cfg)


def _start_evaluation_service_proc(api_cfg, child_env: dict[str, str] | None):
    """Start one service subprocess, optionally with an overridden environment."""
    if not child_env:
        return run_server_proc(api_cfg)
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_run_server_proc_with_env,
        args=(api_cfg, dict(child_env)),
        daemon=True,
    )
    proc.start()
    return proc


def evaluation_worker_api_servers(
    api_server_sets: list[list | None],
) -> list[dict]:
    """Deduplicated worker-scoped service definitions across the given configs.

    The returned dicts still carry their *declared* port; the per-worker offset
    is applied by :func:`start_evaluation_worker_api_servers`.
    """
    return _deduplicated_evaluation_api_servers(api_server_sets, "worker")


def evaluation_session_api_servers(
    api_server_sets: list[list | None],
) -> list[dict]:
    """Deduplicated session-scoped service definitions across the given configs.

    Useful to a caller that wants to report how many session services an
    evaluation adopted versus started.
    """
    return _deduplicated_evaluation_api_servers(api_server_sets, "session")


def start_evaluation_api_servers(
    api_server_sets: list[list | None],
    *,
    wait_timeout: float,
    child_env: dict[str, str] | None = None,
    probe_timeout: float = 10.0,
) -> list:
    """Make the deduplicated SESSION-scoped services available to the evaluation.

    Session services (segmentation, grasp proposal) are stateless across trials
    and dominated by model-load time, so they are normally RESIDENT: started
    once by an external keeper and shared by every evaluation.  This function
    therefore ADOPTS a service that is already listening and starts one only
    when nothing is.  The decision is automatic -- occupancy plus identity --
    so no caller has to remember a flag, and a run still works with no keeper.

    Adoption is identity-checked, never positional: the running server's
    ``/openapi.json`` must expose the routes that service is required to serve
    (:data:`EVALUATION_SERVICE_REQUIRED_ROUTES`, or a per-entry
    ``required_routes``).  Anything else raises.

    Adopted services are deliberately NOT in the returned list, so
    :func:`_stop_api_servers` cannot reach them: an evaluation must never take
    down a resident service it did not start.

    Args:
        api_server_sets: one ``api_servers`` list per selected env config.
        wait_timeout: seconds to wait for the services *this call starts* to
            listen.  Required; this layer never supplies a default.
        child_env: environment overrides applied inside each service
            subprocess this call starts, never in the caller's ``os.environ``.
            It does not reach an adopted service -- its keeper chose its device.
        probe_timeout: seconds allowed for the ``/openapi.json`` identity probe.

    Returns:
        Only the processes this call started, for ``_stop_api_servers``.

    Note:
        There is deliberately no ``allow_existing`` parameter here any more.
        Session scope always adopts a verified resident service.  The
        per-worker path keeps ``allow_existing=False``, because a worker's
        ports must be exclusively its own.
    """
    session_servers = evaluation_session_api_servers(api_server_sets)
    procs: list = []
    pending: list[tuple[str, int, str, Any]] = []
    adopted: list[str] = []
    try:
        for api_server in session_servers:
            host = str(api_server.get("host", "127.0.0.1"))
            declared_port = api_server.get("port")
            role = _evaluation_service_label(api_server)
            port = None if declared_port is None else int(declared_port)
            if port is not None:
                reserved = RESERVED_EVALUATION_PORTS.get(port)
                if reserved is not None:
                    raise RuntimeError(
                        f"Evaluation session service {role} would bind reserved port "
                        f"{port} ({reserved}). Change the declared port."
                    )
                if _evaluation_port_is_open(host, port):
                    pid = _verify_resident_evaluation_service(
                        api_server, host, port, probe_timeout=probe_timeout
                    )
                    owner = f"pid={pid}" if pid is not None else "pid=unknown"
                    print(
                        f"[evaluation-services] ADOPTED {role} at {host}:{port} "
                        f"({owner}) -- resident, identity verified; this evaluation "
                        f"did NOT place it and will NOT stop it"
                    )
                    adopted.append(role)
                    continue
            proc = _start_evaluation_service_proc(
                _evaluation_service_launch_config(api_server), child_env
            )
            procs.append(proc)
            print(
                f"[evaluation-services] STARTED {role} at {host}:{port} -- owned by "
                f"this evaluation and stopped at teardown"
            )
            if port is not None:
                pending.append((host, port, role, proc))
        _wait_for_evaluation_services(pending, wait_timeout)
    except BaseException:
        _stop_api_servers(procs)
        raise
    print(
        f"[evaluation-services] session services ready: {len(adopted)} adopted "
        f"(left running), {len(procs)} started by this evaluation"
    )
    return procs


def start_evaluation_worker_api_servers(
    worker_api_servers: list[dict],
    *,
    worker_slot: int,
    wait_timeout: float,
    allow_existing: bool = False,
) -> tuple[list, dict[str, str]]:
    """Start this worker's own services; return ``(procs, client_env)``.

    Each service binds ``declared_port + worker_slot`` so that N workers get N
    disjoint endpoints.  Every computed port is checked against
    :data:`RESERVED_EVALUATION_PORTS` and against actual occupancy, so a slot
    that would land on another system's port fails loudly instead of producing
    a mysterious downstream error.

    ``client_env`` maps each service's URL environment variable to this
    worker's endpoint; the caller (or :func:`worker_service_session`) is
    responsible for scoping it to the worker's lifetime.
    """
    slot = int(worker_slot)
    if slot < 0 or slot >= MAX_EVALUATION_WORKER_SLOTS:
        raise ValueError(
            f"worker_slot must be in [0, {MAX_EVALUATION_WORKER_SLOTS}), "
            f"got {worker_slot!r}"
        )
    procs: list = []
    pending: list[tuple[str, int, str, Any]] = []
    client_env: dict[str, str] = {}
    allocated: dict[int, str] = {}
    try:
        for api_server in worker_api_servers or []:
            entry = dict(api_server)
            scope = evaluation_api_server_scope(entry)
            if scope != "worker":
                raise ValueError(
                    f"{_evaluation_service_label(entry)!r} is {scope}-scoped and must "
                    f"not be started per worker; pass only the list returned by "
                    f"evaluation_worker_api_servers()."
                )
            declared_port = entry.get("port")
            if declared_port is None:
                raise ValueError(
                    f"Worker-scoped api_server {_evaluation_service_label(entry)!r} "
                    f"declares no port; a per-worker service needs one so each worker "
                    f"slot can be offset from it."
                )
            host = str(entry.get("host", "127.0.0.1"))
            port = int(declared_port) + slot
            role = f"{_evaluation_service_label(entry)} (worker slot {slot})"
            clash = allocated.get(port)
            if clash is not None:
                raise RuntimeError(
                    f"Worker slot {slot} maps two services onto {host}:{port}: "
                    f"{clash} and {role}. Space the declared ports at least "
                    f"{MAX_EVALUATION_WORKER_SLOTS} apart."
                )
            allocated[port] = role
            url_env_name = _evaluation_service_url_env(entry)
            if _claim_evaluation_port(
                host, port, role=role, allow_existing=allow_existing
            ):
                cfg = _evaluation_service_launch_config(entry)
                cfg["port"] = port
                proc = _start_evaluation_service_proc(cfg, None)
                procs.append(proc)
                pending.append((host, port, role, proc))
                print(f"Evaluation worker service {role} started on {host}:{port}")
            else:
                print(
                    f"Evaluation worker service {role} already served on "
                    f"{host}:{port}; adopting it"
                )
            client_env[url_env_name] = f"http://{host}:{port}"
        _wait_for_evaluation_services(pending, wait_timeout)
    except BaseException:
        _stop_api_servers(procs)
        raise
    return procs, client_env


@contextlib.contextmanager
def worker_service_session(
    worker_api_servers: list[dict],
    *,
    worker_slot: int,
    wait_timeout: float,
):
    """Own one worker's services for its lifetime, scoping ``client_env`` to it.

    On exit the previous ``os.environ`` values are restored *before* the
    services are stopped, so nothing can observe a live endpoint variable that
    points at an already-terminated server.
    """
    server_procs, client_env = start_evaluation_worker_api_servers(
        worker_api_servers,
        worker_slot=worker_slot,
        wait_timeout=wait_timeout,
    )
    previous = {key: os.environ.get(key) for key in client_env}
    os.environ.update(client_env)
    try:
        yield client_env
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _stop_api_servers(server_procs)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def _setup_output_dir(args, config: dict[str, Any]) -> None:
    """Create and normalize the output directory path.

    Inserts the model name into the output path so that results from
    different models are stored separately.
    """
    if args.use_oracle_code:
        args.model = "oracle"
    if config["output_dir"]:
        parts = config["output_dir"].split("/")
        parts.insert(-1, str(args.model).replace("/", "_"))
        new_out_dir = "/".join(parts)
        Path(new_out_dir).mkdir(parents=True, exist_ok=True)
        config["output_dir"] = new_out_dir


# ---------------------------------------------------------------------------
# Headless trial runner
# ---------------------------------------------------------------------------

def _run_headless_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Run all trials in headless CLI mode, then print a summary.

    Dispatches to parallel or sequential execution depending on ``num_workers``.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    if config["record_video"] and not config["output_dir"]:
        raise ValueError("record_video requires --output-dir to save generated videos")

    _setup_output_dir(args, config)

    # Determine which trials to run (supports resume)
    trial_ids = list(range(1, config["total_trials"] + 1))
    if config.get("resume_idx") is not None:
        trial_ids = list(range(config["resume_idx"], config["total_trials"] + 1))

    # Execute trials
    if config["num_workers"] > 1:
        setup_fn = functools.partial(_worker_setup, env_factory=env_factory)
        trial_fn = functools.partial(_run_single_trial_worker, args=args, config=config)
        summaries = run_parallel_with_setup(
            trial_ids,
            num_workers=config["num_workers"],
            setup_fn=setup_fn,
            trial_fn=trial_fn,
        )
    else:
        summaries = _run_trial_batch(trial_ids, args=args, env_factory=env_factory, config=config)

    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)

    # Write completion flag
    os.makedirs(os.path.join(config["output_dir"], "aaa_done_flag"), exist_ok=True)
    with open(os.path.join(config["output_dir"], "aaa_done_flag", "aaa_done_flag.txt"), "w") as f:
        f.write("1")


# ---------------------------------------------------------------------------
# Worker setup and trial dispatch
# ---------------------------------------------------------------------------

def _worker_setup(*, env_factory: dict[str, Any]) -> tuple[Any, str | None]:
    """Create the environment once per parallel worker.

    Returns:
        Tuple of (environment, multi_turn_prompt).
    """
    env = instantiate(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)
    return (env, multi_turn_prompt)


def _run_single_trial_worker(
    state: tuple[Any, str | None],
    trial: int,
    *,
    args,
    config: dict[str, Any],
) -> TrialSummary:
    """Run a single trial using pre-initialized worker state (for parallel mode)."""
    env, multi_turn_prompt = state
    return _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)


def _run_trial_with_retries(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
) -> TrialSummary:
    """Attempt a trial up to MAX_TRIAL_RETRIES times, retrying on timeout."""
    for attempt in range(MAX_TRIAL_RETRIES):
        try:
            is_last_attempt = attempt == MAX_TRIAL_RETRIES - 1
            return _run_single_trial_with_timeout(
                env=env,
                trial=trial,
                args=args,
                config=config,
                multi_turn_prompt=multi_turn_prompt,
                timeout_s=TRIAL_TIMEOUT_SECONDS,
                raise_on_timeout=not is_last_attempt,
            )
        except TimeoutError:
            print(f"Trial {trial} timed out (attempt {attempt + 1}/{MAX_TRIAL_RETRIES}). Retrying...")

    # All retries exhausted
    return TrialSummary(
        trial=trial,
        success=False,
        reward=0.0,
        terminated=False,
        truncated=True,
        sandbox_rc=1,
        log=f"Trial {trial} failed after {MAX_TRIAL_RETRIES} timeout retries",
        task_completed=False,
        code_path=None,
        num_regenerations=0,
        num_finishes=0,
        num_code_blocks=0,
    )


def _run_trial_batch(
    trial_ids: Iterable[int],
    *,
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> list[TrialSummary]:
    """Run a batch of trials sequentially (single-worker mode).

    Creates one environment instance and reuses it for all trials.
    """
    trial_indices = list(trial_ids)
    if not trial_indices:
        return []

    # OmniGibson / Isaac Sim reads sys.argv and can crash on unknown flags.
    # Strip our CLI args while instantiating the env.
    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv

    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)

    summaries: list[TrialSummary] = []
    for trial in tqdm(trial_indices, desc="Running Trials"):
        summary = _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)
        summaries.append(summary)

    summaries.sort(key=lambda s: s.trial)
    return summaries


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def _run_single_trial_with_timeout(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    timeout_s: float,
    raise_on_timeout: bool = False,
) -> TrialSummary:
    """Run a single trial with a wall-clock timeout via SIGALRM."""
    timeout_seconds = max(1, int(timeout_s))
    timed_out = False

    def _timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        nonlocal timed_out
        timed_out = True
        raise TimeoutError(f"Trial {trial} exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    partial_artifacts: dict[str, Any] = {}
    try:
        return _run_single_trial(
            env, trial, args, config, multi_turn_prompt, partial_artifacts=partial_artifacts
        )
    except BaseException as exc:
        is_timeout = timed_out or isinstance(exc, TimeoutError)
        try:
            import requests as _requests
            is_timeout = is_timeout or isinstance(exc, _requests.exceptions.Timeout)
        except Exception:
            pass

        if not is_timeout:
            raise
        if raise_on_timeout:
            raise TimeoutError(f"Trial {trial} timed out") from exc

        print(f"Trial {trial} timed out after {timeout_seconds} seconds")
        return _build_timeout_summary(trial, timeout_seconds, partial_artifacts, config, exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_timeout_summary(
    trial: int,
    timeout_seconds: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
) -> TrialSummary:
    """Build a TrialSummary from partial artifacts after a timeout."""
    raw_code = pa.get("raw_code", "")
    code_blocks = pa.get("code_blocks", [])
    code_block_metadata = pa.get("code_block_metadata", [])
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)

    info_step = pa.get("info_step", {"sandbox_rc": 1, "stdout": "", "stderr": str(exc)})
    if info_step.get("stderr") == "":
        info_step["stderr"] = str(exc)
    else:
        info_step["stderr"] += f"\n\nTimeout Error: {exc}"

    reward = pa.get("reward", 0.0)
    terminated = pa.get("terminated", False)
    truncated = pa.get("truncated", False)
    num_regenerations = pa.get("num_regenerations", 0)
    num_finishes = pa.get("num_finishes", 0)
    num_code_blocks = pa.get("num_code_blocks", len(code_blocks))

    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        prefix=f"Trial {trial} timed out after {timeout_seconds} seconds.",
    )

    code_path = _save_trial_artifacts(
        config, trial,
        sandbox_rc=1,
        reward=reward,
        task_completed=info_step.get("task_completed", False),
        final_code=final_code,
        raw_code=raw_code,
        all_responses=pa.get("all_responses", []),
        log_lines=log_lines,
        visual_feedback_imgs=pa.get("visual_feedback_imgs", []),
        ensemble_data=pa.get("ensemble_data"),
        multiturn_ensemble_data=pa.get("multiturn_ensemble_data", []),
    )

    return TrialSummary(
        trial=trial,
        success=False,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=1,
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", False),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )