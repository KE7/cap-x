# cap-x on aarch64 / NVIDIA DGX Spark (GB10) — setup guide

This guide reproduces every cap-x venv on an **aarch64 NVIDIA GB10 / DGX Spark**
box, where the default x86_64 instructions in the top-level README do **not**
apply. It supersedes the ad-hoc `VENV_SETUP_HANDOFF.md` that previously lived
(untracked) at the repo root.

> If you are on a normal x86_64 + CUDA 12.x machine, follow the
> [README Installation](../README.md#installation) instead — this guide is for
> the ARM/Spark path only.

---

## 1. Machine context

| Property | Value |
|---|---|
| OS | Ubuntu 24.04 LTS (noble) |
| Arch | **aarch64** (ARM64) |
| GPU | **NVIDIA GB10** (Grace-Blackwell, DGX Spark), compute capability **sm_121** |
| CUDA toolkit | **13.0** at `/usr/local/cuda` (`nvcc V13.0.88`) |
| Driver | 580.x |
| Python | **3.11** for all venvs (see §3 — open3d & Isaac 5.1 both force 3.11 on this box) |
| uv | 0.9.x ([install](https://docs.astral.sh/uv/)) |

**Why aarch64 changes everything:**
- Default PyPI `torch` on aarch64 is **CPU-only** — you must install the
  **cu130** aarch64 wheels (§4) to get a working GPU.
- `pyproject.toml [tool.uv] environments` is pinned to **x86_64 only**, so
  `uv sync` hard-errors on this box (*"current Python platform is not compatible
  with the lockfile's supported environments: platform_machine == 'x86_64'"*).
  We use a `uv pip install` path with an overrides file instead (§5).
- Two base deps have **no usable aarch64 wheel**: `open3d` (only `0.18.0`, and
  only cp38–cp311 → forces Python **3.11**) and `decord` (none at all → gated
  off; restore via source build, §8).
- The `sm_121` "not supported" warnings printed by some tools are **benign** —
  CUDA 13 JITs from PTX (`TORCH_CUDA_ARCH_LIST=12.1+PTX`).

**Submodules:** ensure they are initialised first:
```bash
git submodule update --init --recursive
```

---

## 2. The four venvs at a glance

| Venv | Path | Purpose | Setup |
|---|---|---|---|
| robosuite | `.venv` | Robosuite (1.5) benchmark | `./scripts/setup_robosuite_venv.sh` (§6) |
| libero | `.venv-libero` | LIBERO-PRO (robosuite 1.4) + contact-graspnet | `./scripts/setup_libero_venv.sh` (§6) |
| pyroki | `.venv-pyroki` | GPU JAX IK/plan HTTP server | `./scripts/setup_pyroki_venv.sh` (§7) † |
| BEHAVIOR | `capx/third_party/b1k/.venv` | OmniGibson + Isaac Sim 5.1 | manual, see §9 (working) |

robosuite (1.5) and LIBERO (robosuite 1.4) declare a conflict, so they need
separate venvs. pyroki runs GPU JAX (numpy 2.x) which is incompatible with the
benchmark venvs' numpy 1.26.4, so it is isolated and reached over HTTP.

> † `scripts/setup_pyroki_venv.sh` and the launcher's per-venv routing currently
> live on branch **`feat/pyroki-gpu-venv`** (PR #2, not yet merged to `main`/this
> docs branch). Until that lands, fetch the pyroki script from that branch first
> (see §7) — the command blocks below will not exist on this branch otherwise.

---

## 3. Why Python 3.11 everywhere

| Constraint | Forces |
|---|---|
| `open3d==0.18.0` is the only open3d with an aarch64 linux wheel covering cp311 (cp38–cp311). 0.19.0 has no aarch64 wheel; **no cp312 wheel at any version**. | benchmark venvs = **3.11** (3.12 would require an open3d source build) |
| Isaac Sim 5.1 (the only Isaac with an aarch64 build, source-built for Spark) bundles **Python 3.11** (kit 107 dropped cp310). | BEHAVIOR venv = **3.11** |

The README's "Requires Python 3.10" line is for the x86 path and is **stale on
this box**.

---

## 4. The shared cu130 torch recipe (all GPU venvs)

Always export this build env **before any build** (the setup scripts do it for
you; do it manually for the BEHAVIOR/curobo steps):

```bash
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.1+PTX"   # GB10 = sm_121; +PTX so CUDA 13 JITs to it
```

Install CUDA-enabled torch from the cu130 aarch64 index:

```bash
uv pip install --python <venv>/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  torch==2.12.0+cu130 torchvision==0.27.0+cu130 torchaudio==2.11.0+cu130
```

The versions are pinned to the verified cu130 aarch64 cp311 wheel set so reruns
are reproducible (the `+cu130` local tag exists only on the pytorch index, so it
also disambiguates from the CPU-only PyPI wheels). Verified result on this box:
**`torch 2.12.0+cu130`, `torch.cuda.is_available() == True`,
device `NVIDIA GB10`** (torchvision 0.27.0, torchaudio 2.11.0).
Source: build.nvidia.com/spark/isaac + DGX Spark porting guides.

---

## 5. The aarch64 overrides file (reproducible pins)

`scripts/overrides/aarch64-overrides.txt` captures the dependency pins the
benchmark venvs need. It exists because `uv pip install` does **not** auto-apply
`pyproject.toml [tool.uv] override-dependencies` (those apply only to
`uv sync`/`uv lock`, which is blocked here). The file mirrors the pyproject
override block **plus** two aarch64-specific pins:

- `open3d==0.18.0` — real aarch64 cp311 wheel, no source build (see §3).
- `decord ; sys_platform == 'never'` — gate it out (no aarch64 wheel anywhere;
  sam3-train-only — restore via source build, §8).

Both `setup_robosuite_venv.sh` and `setup_libero_venv.sh` pass it via
`--overrides`. Keep its first block in sync with `pyproject.toml`.

---

## 6. robosuite (`.venv`) and libero (`.venv-libero`)

One-liners (idempotent; safe to re-run; they recreate the venv):

```bash
./scripts/setup_robosuite_venv.sh    # -> .venv   (robosuite 1.5)
./scripts/setup_libero_venv.sh       # -> .venv-libero  (LIBERO + contact-graspnet)
```

Each script: creates a **py3.11** venv → installs **cu130 torch** (§4) →
`uv pip install --overrides scripts/overrides/aarch64-overrides.txt -e ".[<extra>]"`
→ applies the **curobo lerp patch** (§ below) and builds editable curobo with
`--no-build-isolation` against the cu130 torch → runs an import verify.

Verified final imports (libero venv, real output):
```
torch 2.12.0+cu130 cuda True NVIDIA GB10
open3d 0.18.0
curobo 0.7.8.post1.dev0+dirty
libero OK · robosuite 1.4.0 · contact_graspnet_pytorch OK
decord absent (expected, gated)
```
robosuite venv is the same recipe with the `[robosuite]` extra (verify imports
`torch, robosuite, curobo, open3d`).

### The curobo lerp patch (`patches/curobo-lerp.patch`)

curobo's CUDA extension fails to build under CUDA 13 because C++20's `constexpr
std::lerp(float,float,float)` collides with curobo's identical-signature scalar
overload. The tracked patch `patches/curobo-lerp.patch` guards curobo's overload
out when `std::lerp` is available. The setup scripts apply it automatically; to
apply by hand:

```bash
# Run from the repo root. With `-C` the patch path is resolved from inside the
# submodule, so use the repo-root-relative `../../../` prefix:
git -C capx/third_party/curobo apply ../../../patches/curobo-lerp.patch
# or, without -C:  git apply patches/curobo-lerp.patch --directory=capx/third_party/curobo
# or:  ( cd capx/third_party/curobo && patch -p1 < ../../../patches/curobo-lerp.patch )
```

This is currently a **working-tree edit to the curobo submodule** (not committed
upstream). The patch file is the reproducible capture; consider upstreaming.

---

## 7. pyroki GPU service (`.venv-pyroki`)

> **Note:** the `setup_pyroki_venv.sh` script and the `launch_servers.py`
> per-venv routing live on branch **`feat/pyroki-gpu-venv`** (PR #2) and are not
> yet on `main` (this docs branch). The script does **not** exist on this branch
> — fetch it from that branch first (or run this section after PR #2 merges):

```bash
# Until feat/pyroki-gpu-venv (PR #2) merges, retrieve the script from that branch:
git fetch origin feat/pyroki-gpu-venv
git checkout origin/feat/pyroki-gpu-venv -- scripts/setup_pyroki_venv.sh
# then run it:
./scripts/setup_pyroki_venv.sh
```

Creates `.venv-pyroki` (py3.11) with `jax[cuda13]` (resolves to jax 0.10.1,
numpy 2.x) + pyroki + FastAPI server deps, installed from `/tmp` so the repo's
numpy-1.26.4 / jax<0.4.30 overrides are not applied. Verified GPU:
`jax.devices() == [CudaDevice(id=0)]`; a real `POST /ik` returned valid 8-DOF
joints with **no cuSolver error** (pyroki issue #12 cleared on jax[cuda13]==0.10.1).

**Launch (direct):**
```bash
.venv-pyroki/bin/python -m capx.serving.launch_pyroki_server   # binds 127.0.0.1:8116
```
**Launch (via orchestrator — routes pyroki to `.venv-pyroki` automatically):**
```bash
.venv/bin/python -m capx.serving.launch_servers --profile default
```
Benchmark code reaches it over HTTP via `init_pyroki()` → `http://127.0.0.1:8116`.

**Remote / cross-machine use:**
- Serve on `0.0.0.0` so other hosts can reach it (the server currently binds
  `127.0.0.1` by default; bind-address wiring via argv is a known limitation —
  `main()` does not yet parse `--host/--port`).
- Point clients at it with `PYROKI_SERVER_URL=http://<host>:8116` (or pass the
  URL to `init_pyroki(...)`).

**Deploy tuning:** JAX pre-allocates ~75% of GB10 unified memory (~93 GB) on
start. If co-locating pyroki with the torch servers on this single-GPU box, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`)
to avoid OOM.

---

## 8. decord source build (optional — sam3 training only)

decord has no aarch64 wheel, so it is gated out of the benchmark venvs. LIBERO
eval/train and contact-graspnet inference do **not** need it; only sam3
training/video utilities do. To restore it, do a CPU source build:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev \
  libavcodec-dev libavformat-dev libavfilter-dev libavutil-dev \
  libavdevice-dev libswscale-dev libswresample-dev
git clone --recursive https://github.com/dmlc/decord    # or georgia-tech-db/eva-decord
cd decord && mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release         # CPU build (safe on GB10)
make -j"$(nproc)"
cd ../python && uv pip install --python /ABS/PATH/.venv-libero/bin/python .
```

Similarly, `open3d` on py3.12 (if you ever must use 3.12) has no wheel — build
via the Open3D `openblas-arm64-py312` docker recipe. Avoided entirely by staying
on py3.11.

---

## 9. BEHAVIOR (OmniGibson + Isaac Sim 5.1) — WORKING

> ✅ **Status: working on this GB10 box.** BEHAVIOR runs end-to-end with
> **Isaac Sim 5.1 + OmniGibson 3.8.0 + bddl3 3.8** (b1k branch
> `feat/omnigibson-3.8.0-isaac5.1`, commits `606dcfc40` OmniGibson + `86fae07ab`
> bddl3). The cuRobo motion path that previously appeared blocked is **resolved**
> — R1Pro warmup + plan PASS and a real `turning_on_radio` BEHAVIOR episode ran
> end-to-end on the real GB10 (cap (12,1)), no crash. The fix is **three small
> in-process patches** (below); there was **no CUDA-12.8-toolkit / out-of-process
> / cuMotion requirement**. The b1k submodule work lives on branch
> `feat/omnigibson-3.8.0-isaac5.1` and is owned by a separate worker; this
> section documents the wiring, it does not script it. Remaining gaps are about a
> *fully-scored agent eval* (perception servers, LLM backend), **not** the motion
> fix — see §10.

**Key facts established on this box:**
- The only Isaac Sim with an aarch64 build is **5.1.0**, **source-built** per the
  NVIDIA/ARM DGX Spark guide. It bundles **Python 3.11**. There is **no aarch64
  Isaac 4.5.0**, so the vendored OmniGibson 3.7.2 (Isaac-4.5-only) cannot run —
  it was bumped to **3.8.0** (Isaac-5.1) on the b1k branch (commits `606dcfc40`
  OmniGibson + `86fae07ab` bddl3).
- On OmniGibson 3.8.0 the pymeshlab pin is arch-split, so aarch64 resolves
  **`pymeshlab 2025.7.post1`** automatically (no manual edit needed).

**Isaac source build path (MACHINE-SPECIFIC — confirm per machine):**
```
/home/batman/Documents/open-source/isaacsim/_build/linux-aarch64/release
```

**Wired env to import Isaac + OmniGibson from the b1k venv** (the venv is
separate from the source-built Isaac; both are cp311):
```bash
export ISAAC_PATH=/home/batman/Documents/open-source/isaacsim/_build/linux-aarch64/release
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"
source "$ISAAC_PATH/setup_python_env.sh"     # sets PYTHONPATH + LD_LIBRARY_PATH
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$ISAAC_PATH/kit/libcarb.so"   # libgomp = aarch64 quirk
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 CUDA_HOME=/usr/local/cuda
```
Verified under this env: `import isaacsim` (source build), `import omnigibson` ==
`3.8.0`, and Isaac's bundled `torch 2.7.0+cu128` reports `cuda True`.

**`uv_install.sh` edits (working-tree on the b1k branch, NOT yet committed):**
- `PYTHON_VERSION` 3.10 → **3.11**.
- **Skip** the x86_64 Isaac 4.5 cp310 wheel download/install block — reuse the
  source-built aarch64 Isaac 5.1 instead.
- Relax the guard that aborts when `ISAAC_PATH`/`EXP_PATH` are set; export them.
- Rewire the verify step to `source $ISAAC_PATH/setup_python_env.sh` + the
  `LD_PRELOAD` above; make it non-fatal.

### The cuRobo "illegal memory access" — root-caused and fixed (NOT a CUDA-toolkit issue)

The cuRobo `lbfgs_step_cu` *"illegal memory access"* crash that previously looked
like a hard blocker was **fully root-caused on this GB10 and fixed in-process** —
it was **never** a cu128-vs-cu130, CUDA-context, build-against-Isaac-torch, or
local-toolkit problem. cuRobo runs **in-process fine** under Isaac's bundled
`torch 2.7.0+cu128` (franka and R1Pro base+arm all plan successfully).

**Root cause:** OmniGibson 3.8.0 already excludes the crash-prone R1Pro `default`
cuRobo embodiment on Blackwell, but its guard in
`OmniGibson/omnigibson/action_primitives/curobo.py` only matched GPU capability
`(12,0)` (RTX 50-series). **GB10 is `(12,1)`**, so the guard missed and R1Pro kept
its crashing `default` embodiment → illegal memory access in `lbfgs_step_cu`.

**Fix — three small in-process patches on the b1k branch** (no CUDA 12.8 toolkit,
no out-of-process server, no cuMotion):

1. **Commit `93695c82e`** — widen the Blackwell guard from `== (12,0)` to
   `get_device_capability(device)[0] == 12` so it covers GB10 `(12,1)` (and all
   future sm_12x). This **excludes** R1Pro's `default` embodiment and keeps its
   safe **base + arm** embodiments (and keeps Tiago's `default`). Applied at the
   functional site in `action_primitives/curobo.py` plus the `tests/test_curobo.py`
   and `tests/test_primitives.py` guards.
2. **Commit `c5182e88f`** — two follow-on fixes that are real GB10 blockers the
   cu128 runtime hits (the cu130-based RCA never did):
   - **Obstacle-update DEFAULT fallback:** `update_obstacles()`/`remove_obstacles()`
     hard-coded `self.mg[DEFAULT].update_world()`, which `KeyError`s once `default`
     is excluded; the shared world-collision checker now updates via any present
     embodiment.
   - **cu128 NVRTC fuser disable:** under Isaac's cu128 torch on sm_121, the
     TensorExpr JIT fuser emits an NVRTC kernel whose `-arch` cu128 rejects,
     crashing at `import curobo.wrap.reacher.motion_gen`. Disabled the GPU JIT
     fuser on Blackwell sm_12x (runtime flag only — no torch swap / toolkit change).

> ⚠️ **Do NOT set `use_default_embodiment_only=True`.** This was an earlier
> (wrong) idea — it does the *opposite* of the fix: it re-adds R1Pro's crashing
> `default` embodiment and the illegal-memory-access returns. Leave it at its
> default `False`.

**Verified on the real GB10** (cap `(12,1)`, no faked capability, in-process under
Isaac + GPU physics):
- R1Pro `warmup()` + `compute_trajectories` plan → `WARMUP_OK`,
  `PLAN_SUCCESS: True` (kept embodiments `['arm', 'base']`, `default` excluded).
- A real **`turning_on_radio`** BEHAVIOR episode ran end-to-end: scene
  `house_double_floor_lower` built, R1Pro loaded and stepped, R1Pro executed a
  cuRobo-planned **31-waypoint** trajectory (the exact L-BFGS/IK path that used to
  crash) — no illegal-memory crash.
- Dataset in place: BEHAVIOR-1K assets (33 GB) + 2025 challenge task instances
  (400 MB) downloaded to `capx/third_party/b1k/datasets`.

---

## 10. Known gaps / TODO

1. **BEHAVIOR curobo — RESOLVED (no longer a blocker).** Root-caused to the
   OmniGibson Blackwell embodiment guard missing GB10 `(12,1)`; fixed in-process
   on the b1k branch (commits `93695c82e` + `c5182e88f`, see §9). R1Pro
   warmup+plan and a real `turning_on_radio` episode are verified end-to-end on the
   real GB10 — **no CUDA 12.8 toolkit / out-of-process server needed**. The
   remaining gaps below are for a *fully-scored agent eval*, **not** the motion fix:
   - **Perception servers not yet in the b1k venv.** SAM3 + ContactGraspNet are
     installed only in `.venv-libero`, not `capx/third_party/b1k/.venv`; the oracle
     grasp/segmentation path needs them. Install without the curobo reinstall
     (which would undo the fix) and without `decord` (no aarch64 wheel, §8).
   - **aarch64 instance-segmentation render crash (separate bug).** An unrelated
     `OgnInstanceSegmentation::compute` segfault in `omni.replicator.core` on
     camera/`seg_instance` obs modalities (avoided by running proprio-only); a
     full perception-driven trial on this box would need this addressed. Not curobo.
   - **LLM backend for non-oracle configs.** Point the agent at the local Qwen
     vLLM at `:8000` (`Qwen/Qwen3.6-27B-FP8`) or OpenRouter; oracle configs
     (`use_oracle_code: true`) need no LLM key.
2. **Machine-specific Isaac path** (`/home/batman/Documents/open-source/isaacsim/...`)
   is hard-coded to this box — parameterize via `ISAAC_PATH` and confirm per machine.
3. **`uv_install.sh` edits uncommitted** on the b1k branch; committing them (and a
   BEHAVIOR setup script) is pending coordination with the b1k owner. (The curobo
   fix itself is done and committed — see §9 commits `93695c82e` + `c5182e88f`.)
4. **Durable lock (eventual reproducibility PR).** The clean long-term fix is to
   add `aarch64` to `pyproject.toml [tool.uv] environments` and move the aarch64
   pins (open3d, decord gate) into `override-dependencies`, then `uv lock`. That
   touches shared `pyproject.toml`/`uv.lock` and is out of scope for this docs
   branch. The overrides file + scripts here are the interim reproducible path.
5. **pyroki bind address** — the server binds `127.0.0.1` and `main()` ignores
   `--host/--port`; needs argv wiring for non-default ports / remote serving.
6. **decord & open3d-on-py3.12** require source builds (§8) if ever needed.
