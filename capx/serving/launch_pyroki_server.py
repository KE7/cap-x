from __future__ import annotations

import os

# The pyroki IK/plan server only needs capx.integrations.motion.pyroki_snippets,
# not the heavy franka/libero API registry (robosuite/torch/open3d). Default the
# isolation flag ON so this server imports cleanly in its dedicated GPU venv
# (.venv-pyroki) which deliberately omits those deps. setdefault keeps any
# explicit override from the environment. Must run before importing
# capx.integrations.* below.
os.environ.setdefault("CAPX_PYROKI_SERVER_ONLY", "1")

# Disable XLA preallocation BEFORE importing jax (pulled in transitively by
# `pyroki`). This server shares the unified-memory GB10 GPU with the OmniGibson /
# Isaac eval (~30 GB) and the other perception servers; preallocating ~75% of VRAM
# would starve them. With platform allocator + no-prealloc, the IK server's JAX
# footprint stays ~0.6 GB. setdefault preserves any explicit override.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import asyncio
import functools
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import numpy as np
import jax
import jax.numpy as jnp
import pyroki as pk  # type: ignore
import tyro
import uvicorn
import yourdfpy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp

import capx.integrations.motion.pyroki_snippets as pks


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two quaternions.

    Args:
        q_start: Start quaternion in wxyz format
        q_end: End quaternion in wxyz format
        num_steps: Number of interpolation steps

    Returns:
        Array of shape (num_steps, 4) with interpolated quaternions in wxyz format
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    # Create slerp interpolator
    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    # Convert back to wxyz format
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 25,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Args:
        robot: PyRoKi robot model
        target_link_name: Name of the end-effector link
        start_pos: Start position (3,)
        start_wxyz: Start orientation quaternion in wxyz format (4,)
        end_pos: End position (3,)
        end_wxyz: End orientation quaternion in wxyz format (4,)
        num_waypoints: Number of waypoints in the trajectory
        use_prev_cfg: If True, use previous IK solution to bias next solve
        jump_threshold: Max allowed joint change (radians) between waypoints before warning

    Returns:
        Array of shape (num_waypoints, num_joints) with joint configurations
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(15):
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                print(
                    f"Error: {np.linalg.norm(cfg - prev_cfg)}",
                    np.allclose(cfg, prev_cfg, atol=1e-3),
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped {np.degrees(joint_diff[joint_idx]):.1f}° "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    # Print warnings summary
    if jump_warnings:
        print(
            f"\nWARNING: {len(jump_warnings)} large joint jump(s) detected (threshold: {np.degrees(jump_threshold):.1f}°):"
        )
        for warning in jump_warnings:
            print(warning)
        print()

    return np.array(trajectory)


# =====================================================
# Logging and FastAPI app
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None
# Selected compute device. The pyroki IK/plan server runs on GPU-enabled JAX
# (jax[cuda13]); JAX binds its platform at import time and selects the GPU from
# the CUDA_VISIBLE_DEVICES env var that the unified launcher sets per process.
# This global mirrors the sam3/graspnet servers' `_DEVICE` for parity and is set
# from the --device CLI flag the launcher emits for GPU-required servers.
_DEVICE: str = "cuda"

# =====================================================
# R1Pro (/ik_rest) — custom-URDF rest-cost IK
# =====================================================
# The R1Pro BEHAVIOR eval (capx/integrations/r1pro/control.py, the
# CAPX_PYROKI_REMOTE_IK path) POSTs to /ik_rest with a custom R1Pro URDF and
# pks.solve_ik_rest (18-DOF dual-arm). This server now serves BOTH the generic
# panda /ik + /plan AND the R1Pro /ik_rest on the same port so a single GPU
# pyroki service covers the generic motion clients and the R1Pro eval.
#
# The URDF is committed alongside this server (assets/r1pro_ik.urdf). Only the
# kinematic tree matters for solve_ik_rest, so it is loaded with
# load_meshes=False — this makes the server self-contained (no dependency on the
# b1k submodule's mesh tree) while producing identical kinematics to the
# original asset (verified: 18 actuated joints, identical link names).
_DEFAULT_R1PRO_URDF = os.environ.get(
    "CAPX_PYROKI_R1PRO_URDF",
    str(Path(__file__).parent / "assets" / "r1pro_ik.urdf"),
)


@lru_cache(maxsize=4)
def _get_r1pro_robot(urdf_path: str) -> pk.Robot:
    logger.info(f"Loading R1Pro URDF for /ik_rest: {urdf_path}")
    # Meshes are not needed for kinematic IK; skip them so the server does not
    # depend on the URDF's relative mesh/ tree.
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False)
    robot = pk.Robot.from_urdf(urdf)
    logger.info(
        f"R1Pro robot loaded: {robot.joints.num_actuated_joints} actuated joints; "
        f"gripper links present: {[n for n in robot.links.names if 'gripper_link' in n]}"
    )
    return robot


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking GPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request with optional prev_cfg for vel-cost IK."""

    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None  # optional


class IkResponse(BaseModel):
    joint_positions: list[float]


class IkRestRequest(BaseModel):
    """R1Pro rest-cost IK request (mirrors control.py solve_ik_rest call)."""

    target_link_name: str
    target_position: list[float]  # (3,)
    target_wxyz: list[float]  # (4,) wxyz
    rest_cost_weights: list[float] | None = None
    initial_q: list[float] | None = None
    urdf_path: str | None = None


class IkRestResponse(BaseModel):
    cfg: list[float]


class ObstacleEntry(BaseModel):
    type: str
    point: list[float] | None = None
    normal: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    position: list[float] | None = None
    height: float | None = None
    extent: list[float] | None = None  # for box obstacles


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    end_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    obstacles: list[dict] | None = None  # optional list of ObstacleEntry
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float


# =====================================================
# INTERNAL HELPERS
# =====================================================


def _build_world_coll(obstacles: list[dict[str, Any]] | None):
    if obstacles is None:
        return []
    world = []
    for obj in obstacles:
        t = obj.get("type")
        if t == "halfspace":
            p = np.array(obj["point"], dtype=np.float64)
            n = np.array(obj["normal"], dtype=np.float64)
            world.append(pk.collision.HalfSpace.from_point_and_normal(p, n))
        elif t == "sphere":
            c = np.array(obj["center"], dtype=np.float64)
            r = float(obj["radius"])
            world.append(pk.collision.Sphere.from_center_and_radius(c, np.array([r])))
        elif t == "capsule":
            pos = np.array(obj["position"], dtype=np.float64)
            rad = float(obj["radius"])
            h = float(obj["height"])
            world.append(
                pk.collision.Capsule.from_radius_height(
                    position=pos,
                    radius=np.array([rad]),
                    height=np.array([h]),
                )
            )
        elif t == "box":
            extent = np.array(obj["extent"], dtype=np.float64)
            pos = np.array(obj.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            world.append(
                pk.collision.Box.from_extent(
                    extent=extent,
                    position=pos,
                )
            )
        else:
            logger.warning(f"Unknown obstacle type '{t}', ignoring")
    return world


# =====================================================
# SERVER INIT — Load Pyroki Only Once
# =====================================================
def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


def init_pyroki_server(
    robot_urdf_name: str = "panda_description", target_link_name: str = "panda_hand"
):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK

    logger.info(f"Loading robot URDF '{robot_urdf_name}' with Pyroki...")

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = set_min_distance_from_limits(load_robot_description(robot_urdf_name))

    _ROBOT = pk.Robot.from_urdf(urdf)
    # _ROBOT_COLL = pk.collision.RobotCollision.from_urdf(urdf)
    sphere_decomposition = json.load(open(Path(__file__).parent / "assets" / "panda_spheres.json"))
    _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    _TARGET_LINK = target_link_name

    logger.info("PyRoki loaded and ready!")


# =====================================================
# ROUTES
# =====================================================


def _do_solve_ik(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None) -> list[float]:
    """Blocking IK solve (GPU-bound)."""
    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
            prev_cfg=prev_cfg,
        )
    return list(map(float, q))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target_pose_wxyz_xyz = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None

    try:
        joints = await _run_in_thread(_do_solve_ik, target_pose_wxyz_xyz, prev_cfg)
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(joint_positions=joints)


def _do_solve_ik_rest(req: IkRestRequest) -> list[float]:
    """Blocking R1Pro rest-cost IK solve (GPU-bound).

    Mirrors capx/integrations/r1pro/control.py's in-process call
    `pks.solve_ik_rest(robot, target_link_name, target_position, target_wxyz,
    rest_cost_weights, initial_q)` so the GPU result is the same algorithm as the
    CPU path it replaces.
    """
    robot = _get_r1pro_robot(req.urdf_path or _DEFAULT_R1PRO_URDF)
    n = robot.joints.num_actuated_joints
    target_position = np.asarray(req.target_position, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(req.target_wxyz, dtype=np.float64).reshape(4)
    if req.rest_cost_weights is not None:
        rest_cost_weights = jnp.asarray(req.rest_cost_weights, dtype=jnp.float32)
    else:
        rest_cost_weights = 0.0
    initial_q = (
        np.asarray(req.initial_q, dtype=np.float64) if req.initial_q is not None else None
    )
    cfg = pks.solve_ik_rest(
        robot=robot,
        target_link_name=req.target_link_name,
        target_position=target_position,
        target_wxyz=target_wxyz,
        rest_cost_weights=rest_cost_weights,
        initial_q=initial_q,
    )
    cfg = np.asarray(cfg, dtype=np.float64).reshape(-1)
    assert cfg.shape == (n,), f"cfg shape {cfg.shape} != ({n},)"
    return cfg.tolist()


@app.post("/ik_rest", response_model=IkRestResponse)
async def solve_ik_rest(req: IkRestRequest):
    try:
        cfg = await _run_in_thread(_do_solve_ik_rest, req)
    except Exception as e:
        logger.exception("IK rest solve failed")
        raise HTTPException(500, f"IK rest solve failed: {e}")
    return IkRestResponse(cfg=cfg)


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    """Blocking motion planning (GPU-bound)."""
    start_pose = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)

    start_wxyz = start_pose[:4]
    start_position = start_pose[4:]
    end_wxyz = end_pose[:4]
    end_position = end_pose[4:]

    timesteps = req.timesteps
    dt = req.dt

    sol_traj = plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=_TARGET_LINK,
        start_pos=start_position,
        start_wxyz=start_wxyz,
        end_pos=end_position,
        end_wxyz=end_wxyz,
        num_waypoints=timesteps,
    )
    sol_traj = np.asarray(sol_traj)

    return PlanResponse(
        waypoints=sol_traj.tolist(),
        dt=float(dt),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")

    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "jax_devices": [str(d) for d in jax.devices()],
        "robot_loaded": _ROBOT is not None,
        "r1pro_loaded": _get_r1pro_robot.cache_info().currsize > 0,
    }


# =====================================================
# ENTRYPOINT
# =====================================================


def _warm_r1pro():
    """Load the R1Pro robot + JIT-compile solve_ik_rest once at startup so the
    first /ik_rest call from the eval is fast (control.py uses a 60 s timeout)."""
    try:
        robot = _get_r1pro_robot(_DEFAULT_R1PRO_URDF)
        n = robot.joints.num_actuated_joints
        logger.info("Warming R1Pro /ik_rest JIT (left + right gripper)...")
        for link in ("left_gripper_link", "right_gripper_link"):
            pks.solve_ik_rest(
                robot=robot,
                target_link_name=link,
                target_position=np.array([0.4, 0.2, 0.9]),
                target_wxyz=np.array([0.0, 0.0, 1.0, 0.0]),
                rest_cost_weights=jnp.ones(n, dtype=jnp.float32),
                initial_q=np.zeros(n),
            )
        logger.info("R1Pro /ik_rest warm.")
    except Exception:  # noqa: BLE001
        logger.exception("R1Pro warm-up failed (will load lazily on first request)")


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    port: int = 8116,
    host: str = "127.0.0.1",
    device: str = "cuda",
):
    global _DEVICE

    # Record the requested device. The launcher conveys GPU selection for this
    # jax[cuda13] server via CUDA_VISIBLE_DEVICES (set per-subprocess) and passes
    # --device cuda for parity with the other GPU servers; JAX has already bound
    # the visible GPU at import time, so we surface the choice rather than
    # re-binding it here.
    _DEVICE = device

    logger.info(
        "Starting pyroki server on %s:%d (device=%s, robot=%s, target_link=%s)",
        host,
        port,
        device,
        robot,
        target_link,
    )

    # Generic panda /ik + /plan.
    init_pyroki_server(robot_urdf_name=robot, target_link_name=target_link)
    # R1Pro /ik_rest (warm so the eval's first solve is within its timeout).
    _warm_r1pro()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
