from __future__ import annotations

from collections.abc import Callable
from typing import Any

import os

import numpy as np
import requests
from scipy.spatial.transform import Rotation as SciRotation

from capx.utils.serve_utils import post_with_retries

DEFAULT_URL = "http://127.0.0.1:8116"


def init_pyroki(
    server_url: str | None = None,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.

    Downstream code calling ik_solve_fn() or plan_fn() works identically.
    """

    # Resolve PYROKI_SERVICE_URL at CALL time, not import time. PyRoki is
    # WORKER-scoped: worker_service_session() starts one server per slot on
    # declared_port + slot and injects PYROKI_SERVICE_URL into os.environ for
    # that worker's lifetime -- which happens long after this module is
    # imported. Reading it at import would capture the unset value and every
    # solve_ik would hit the dead DEFAULT_URL, which is what made the evolving
    # agent reimplement IK by hand instead of using this service.
    if server_url is None:
        server_url = os.environ.get("PYROKI_SERVICE_URL", DEFAULT_URL)

    # Normalize trailing slash
    server_url = server_url.rstrip("/")

    # =====================================================
    # IK SOLVER WRAPPER
    # =====================================================
    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Identical signature to the local solver, but sends JSON to /ik.
        """

        payload = {
            "target_pose_wxyz_xyz": target_pose_wxyz_xyz.tolist(),
            "prev_cfg": prev_cfg.tolist() if prev_cfg is not None else None,
        }

        data = post_with_retries(f"{server_url}/ik", payload, timeout_seconds=15.0)
        joints = np.asarray(data["joint_positions"], dtype=np.float32)

        return joints

    return ik_solve_fn

    # =====================================================
    # PLANNING WRAPPER
    # =====================================================

def init_pyroki_trajopt(
    server_url: str = DEFAULT_URL,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki_trajopt(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.
    """

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        """
        Identical signature to the local solver, but sends JSON to /plan.
        """
        payload = {
            "start_pose_wxyz_xyz": start_pose_wxyz_xyz.tolist(),
            "end_pose_wxyz_xyz": end_pose_wxyz_xyz.tolist(),
        }
        data = post_with_retries(f"{server_url}/plan", payload)
        waypoints = np.asarray(data["waypoints"], dtype=np.float32)
        return waypoints

    return trajopt_plan_fn

    # def plan_fn(
    #     q_start: np.ndarray,
    #     q_goal: np.ndarray,         # ignored — kept only for API-compat
    #     obstacles: list[dict[str, Any]],
    # ) -> dict[str, Any]:
    #     """
    #     Matches your original signature even though the PyRoKi server ignores `q_goal`.

    #     Returns:
    #         { "waypoints": [...], "dt": float }
    #     """

    #     payload = {
    #         "q_start": np.asarray(q_start, dtype=np.float64).tolist(),
    #         "obstacles": obstacles,
    #     }

    #     data = _post_with_retries(f"{server_url}/plan", payload)

    #     return {
    #         "waypoints": data["waypoints"],
    #         "dt": data["dt"],
    #     }

    # # =====================================================
    # # TRAJECTORY EXECUTION (stub)
    # # =====================================================
    # def exec_traj_fn(traj: dict[str, Any]) -> bool:
    #     return True

    # =====================================================
    # Register identical API hooks
    # =====================================================
    # register_ik(ik_solve_fn)
    # register_motion_planner(plan_fn, exec_traj_fn)
