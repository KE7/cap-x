# cap-x on aarch64 / NVIDIA DGX Spark (GB10) — setup guide

This guide reproduces every cap-x venv on an **aarch64 NVIDIA GB10 / DGX Spark**
box, where the default x86_64 instructions in the top-level README do **not**
apply. It supersedes the ad-hoc `VENV_SETUP_HANDOFF.md` that previously lived
(untracked) at the repo root.

> If you are on a normal x86_64 + CUDA 12.x machine, follow the
> [README Installation](../README.md#installation) instead — this guide is for
> the ARM/Spark path only.
>
> **Note (BEHAVIOR stack):** Isaac Sim **5.1** / OmniGibson **3.8.0** / Python
> **3.11** is now the unified BEHAVIOR stack on **both** architectures. The only
> per-arch difference is how Isaac Sim 5.1 is obtained: **x86_64** installs the
> published cp311 wheels (`isaacsim[all,extscache]==5.1.0` from pypi.nvidia.com,
> handled automatically by `b1k/uv_install.sh`), while **aarch64 / Spark**
> **source-builds** Isaac Sim 5.1 (no aarch64 wheels are published) and reuses it
> via `$ISAAC_PATH`. This guide is the aarch64 source-build deep-dive; x86_64 users
> get 5.1 from wheels per the README and do not need the source build below.

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

The README's "Requires Python 3.10" line is for the x86_64 **base / Robosuite**
venv (which legitimately stays on 3.10); it is **stale on this box**, where every
venv — including the base venv — is **3.11**. Note the BEHAVIOR venv is **3.11 on
both** architectures now (Isaac Sim 5.1 / OG 3.8.0 are cp311 everywhere); only the
x86_64 base/Robosuite venv remains 3.10.

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
→ installs curobo via `scripts/install_curobo.sh` (**prefers the prebuilt wheel**,
falls back to a source build — § below) → runs an import verify.

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

### curobo install: prebuilt wheel, then source build (`scripts/install_curobo.sh`)

curobo's CUDA extension takes **~20-40 min of nvcc** to compile on the Spark, and
it fails to build under CUDA 13 out of the box because C++20's `constexpr
std::lerp(float,float,float)` collides with curobo's identical-signature scalar
overload. Both problems are now solved upstream of this repo:

- The **fork** [`KE7/curobo`](https://github.com/KE7/curobo) (submodule, branch
  `aarch64/cuda13-lerp-fix`, PR #1) carries the std::lerp guard **committed** in
  `helper_math.h` — so no working-tree patch is needed, and
  `patches/curobo-lerp.patch` is **dropped**.
- A **prebuilt wheel** for this platform (cp311 / linux_aarch64 / CUDA 13 /
  sm_121, torch 2.12.0+cu130) is published as a GitHub release on the fork:
  <https://github.com/KE7/curobo/releases/tag/v-cu13-aarch64-30eafef>

`scripts/install_curobo.sh <venv>` (called by the setup scripts) implements the
**prebuilt-wheel-then-build** pattern: it downloads and installs the prebuilt
wheel when the platform matches (skipping the nvcc build), and transparently
falls back to a from-source build of the fork otherwise. Escape hatches:

```bash
# Default: prefer the prebuilt wheel, fall back to source.
scripts/install_curobo.sh .venv-libero

# Force a from-source build (e.g. local curobo edits):
CUROBO_FORCE_SOURCE=1 scripts/install_curobo.sh .venv-libero

# Point at a different prebuilt wheel:
CUROBO_WHEEL_URL=https://github.com/KE7/curobo/releases/download/<tag>/<wheel> \
  scripts/install_curobo.sh .venv-libero
```

To rebuild the prebuilt wheel for a new fork commit: build from a clean checkout
with `TORCH_CUDA_ARCH_LIST=12.1+PTX` + CUDA 13 (`python -m pip wheel . --no-build-isolation`),
then `gh release create v-cu13-aarch64-<shortsha> --repo KE7/curobo <wheel>`.

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
> requirement**. The b1k submodule work lives on branch
> `feat/omnigibson-3.8.0-isaac5.1` and is owned by a separate worker; this
> section documents the wiring, it does not script it. Remaining gaps are about a
> *fully-scored agent eval* (perception servers, LLM backend), **not** the motion
> fix — see §11.

**Key facts established on this box:**
- The only Isaac Sim with an aarch64 build is **5.1.0**, **source-built** per the
  NVIDIA/ARM DGX Spark guide. It bundles **Python 3.11**. There is **no aarch64
  Isaac 4.5.0**, so the vendored OmniGibson 3.7.2 (Isaac-4.5-only) cannot run —
  it was bumped to **3.8.0** (Isaac-5.1) on the b1k branch (commits `606dcfc40`
  OmniGibson + `86fae07ab` bddl3).
- On OmniGibson 3.8.0 the pymeshlab pin is arch-split, so aarch64 resolves
  **`pymeshlab 2025.7.post1`** automatically (no manual edit needed).

**Isaac source build path (per-machine VALUE — set/confirm per box):**
```
/path/to/isaacsim/_build/linux-aarch64/release
```
The *value* is intrinsically machine-specific, but the *entry point is
parameterized*: `b1k/uv_install.sh` (KE7 PR #3) resolves it from `$ISAAC_PATH`
first, else autodetects it repo-relative — you only set the value, nothing is
hard-coded.

**Wired env to import Isaac + OmniGibson from the b1k venv** (the venv is
separate from the source-built Isaac; both are cp311):
```bash
export ISAAC_PATH=/path/to/isaacsim/_build/linux-aarch64/release
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"
source "$ISAAC_PATH/setup_python_env.sh"     # sets PYTHONPATH + LD_LIBRARY_PATH
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$ISAAC_PATH/kit/libcarb.so"   # libgomp = aarch64 quirk
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 CUDA_HOME=/usr/local/cuda
```
Verified under this env: `import isaacsim` (source build), `import omnigibson` ==
`3.8.0`, and Isaac's bundled `torch 2.7.0+cu128` reports `cuda True`.

**`uv_install.sh` aarch64 port — now committed in [`KE7/b1k` PR #3](https://github.com/KE7/b1k/pull/3)**
(branch `fix/aarch64-uv-install-isaac51-source-build`, head `61d8af9e`; OPEN, not
yet merged — cap-x's b1k submodule still pins `qingh097/b1k@272ec5ca`, so the
submodule repoint is the remaining step, tracked by cap-x PR #6
`build/submodule-repoint-ke7-forks`). What the port does:
- `PYTHON_VERSION` 3.10 → **3.11** (Config block).
- Arch-split: on **aarch64**, **skip** the x86_64 Isaac Sim 5.1 cp311 wheel
  download (`isaacsim[all,extscache]==5.1.0` from pypi.nvidia.com) and **reuse**
  the source-built aarch64 Isaac 5.1 instead.
- **`ISAAC_PATH` is parameterized** (not hard-coded): resolved from the
  environment first (preferred, fully overridable), else **repo-relative
  autodetect** at `$WORKDIR[/..]/isaacsim/_build/linux-aarch64/release`, else a
  clear error with guidance. `EXP_PATH`/`CARB_APP_PATH` are derived from it. The
  legacy "abort if Isaac env vars are set" guard now applies **only** to the
  x86_64 wheel path.
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
no out-of-process server):

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

> **Validated-task scope (honest):** `turning_on_radio` is the BEHAVIOR task whose
> cuRobo motion-gen path was driven **end-to-end** on the real GB10 (the 31-waypoint
> trajectory above). `picking_up_trash` — config present at
> `env_configs/r1pro/r1pro_pick_up_trash{,_oracle,_multiturn_vdm}.yaml` — has **not
> yet** had a motion-gen E2E run on this box; it is **deferred**, not validated. The
> in-process cuRobo fix is task-agnostic (it lives in the embodiment guard, not the
> task), so `picking_up_trash` is expected to work, but that is **unverified** here.

---

## 10. Running the benchmarks on Spark (operational runbook)

This is the "how to run it correctly" section. The setup sections above build the
venvs; this one is the rules + commands for an actual scored run on the GB10 head.

### 10.1 Topology — keep the LLM OFF the head

This is a **single-GPU box with 121 GB unified memory**. The simulator (Isaac),
the perception servers, and the IK server all create CUDA contexts on the head
GPU. **A co-located vLLM/LLM will OOM the head.** Keep the policy LLM on the peer
Sparks.

| Component | Where it runs |
|---|---|
| **vLLM / LLM policy server** | **PEER Sparks** (cluster, off the head) |
| Perception (SAM3, ContactGraspNet) | **HEAD** |
| pyroki IK server | **HEAD** (GPU JAX) |
| Isaac Sim / OmniGibson + eval harness | **HEAD** |

Bring the peer-hosted LLM up **only** via the documented `spark-vllm-docker` scripts
(`hf-download.sh` / `run-recipe.sh` / `launch-cluster.sh`) — README at
`/path/to/spark-vllm-docker` (+ `recipes/README.md`); do
**not** hand-roll `docker run` / `vllm serve` / `ray`. Then point the eval at it with
`--server-url` (OpenAI-compatible endpoint):

```bash
--model <served-model-name> --server-url http://<peer-host>:8000/v1/chat/completions
```

Oracle configs (`use_oracle_code: true`) need **no** LLM at all — they run fully on
the head (§10.5). **Rule: never start a vLLM/LLM on the head while an eval is running.**

### 10.2 Head servers — ports must not collide; pyroki is launched SEPARATELY

The eval's `api_servers:` block (see [docs/configuration.md](configuration.md#perception-servers-api_servers))
and `launch_servers.py --profile` auto-launch head servers and **skip any port
already in use** (so you may pre-launch and share across runs). Defaults below are
**examples, not mandates**; the only hard requirement is **no port collisions**:

| Server | Module | Default port (example) | Venv |
|---|---|---|---|
| SAM3 | `capx.serving.launch_sam3_server` | 8114 | `.venv-libero` |
| ContactGraspNet | `capx.serving.launch_contact_graspnet_server` | 8115 | `.venv-libero` |
| pyroki IK | `capx.serving.launch_pyroki_server` | 8116 | **`.venv-pyroki` (GPU) — pre-launch ONLY** |

> ⚠️ **Do NOT let the eval YAML / `launch_servers.py` auto-start pyroki.** On this
> branch the orchestrator spawns every server with `sys.executable` (no per-venv
> routing — that lives on `feat/pyroki-gpu-venv`, PR #2), and the `default`/`full`
> profiles include pyroki. Auto-started from an eval venv, pyroki therefore runs
> under the **eval's CPU-only JAX** (e.g. b1k) → the IK CPU **hang** §10.3 warns
> about. **SAM3 + ContactGraspNet** may be auto-started/shared; **pyroki must be
> pre-launched by hand on `.venv-pyroki`** so it gets the GPU `jax[cuda13]`.

Pre-launch SAM3 + ContactGraspNet (these two only — let the YAML reuse them by port):

```bash
# SAM3 needs the decord stub on PYTHONPATH on aarch64 (no aarch64 decord wheel —
# env-specific shim, NOT needed on a clean x86 install); see §8 + perception notes.
PYTHONPATH=/path/to/sam3_stubs \
  .venv-libero/bin/python -m capx.serving.launch_sam3_server            --device cuda --port 8114 --host 127.0.0.1
.venv-libero/bin/python -m capx.serving.launch_contact_graspnet_server --device cuda --port 8115 --host 127.0.0.1
```

> ⚠️ **Stock `launch_pyroki_server` does not cleanly serve R1Pro on `.venv-pyroki`
> as-is** (both verified on this box): (1) `import capx.serving.launch_pyroki_server`
> under `.venv-pyroki` currently fails with `ModuleNotFoundError: No module named
> 'gymnasium'` (it pulls the full `capx.integrations`/`capx.envs` import chain);
> (2) its default URDF is `panda_description` / `panda_hand` — it serves a **Franka**
> `/ik` route, not R1Pro. **The R1Pro GPU-pyroki launch mechanism is being finalized**
> in the active oracle/fix loop — do not assert an R1Pro launch command here yet
> (pending validation).

> **KEY RULE — one pyroki server PER ROBOT FAMILY.** A pyroki IK server is built for
> **one robot's URDF**. R1Pro and Franka are different robots → **separate** pyroki
> servers, **never** shared. The stock launcher's default URDF is Franka/panda, so a
> Franka eval (Robosuite/LIBERO) gets a Franka `/ik`; an **R1Pro pyroki on the same
> port has no Franka `/ik` route** (the exact collision we hit). Run each family's
> pyroki on a **distinct port** (e.g. Franka 8116, R1Pro 8126 — numbers arbitrary,
> requirement is no collision) and point each eval at its family's port.

### 10.3 pyroki MUST use the GPU JAX venv

pyroki runs in its **own** venv `.venv-pyroki`, which has the **GPU** `jax[cuda13]`
(jaxlib → `CudaDevice`). Reach it over HTTP — do **not** import/run IK in-process
inside the benchmark venvs. (The R1Pro→GPU-pyroki launch/routing specifics are still
under validation — see §10.2; treat as **pending**.)

> ⚠️ **Never run IK in-process in the b1k venv.** Isaac's bundled JAX there is
> **CPU-only** → IK falls back to CPU and **hangs** (frozen-log stall). Always go
> through the pyroki HTTP server backed by `.venv-pyroki`.

JAX pre-allocates ~75% of unified memory on start. When co-locating pyroki with the
torch/perception servers on this single GPU, cap it so the head doesn't OOM:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # or XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
```

### 10.4 BEHAVIOR / Isaac run wiring (head)

BEHAVIOR runs from the b1k venv against the **source-built** aarch64 Isaac Sim 5.1
(§9). Export this env before `launch.py` (Isaac path is **machine-specific** —
confirm per box). The **PYTHONPATH shims are mandatory**: prepend the venv's
`websockets` and `typing_extensions` so they win over Isaac's older bundled copies
(Isaac's prebundled `typing_extensions` lacks `Sentinel`; without the prepend,
imports break).

```bash
export ISAAC_PATH=/path/to/isaacsim/_build/linux-aarch64/release
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"
source "$ISAAC_PATH/setup_python_env.sh"        # sets PYTHONPATH + LD_LIBRARY_PATH
# shims FIRST so the venv copies win over Isaac's bundled ones:
export PYTHONPATH="$WS_SHIM:$TE_SHIM:$PYTHONPATH"   # WS_SHIM=venv websockets, TE_SHIM=venv typing_extensions
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$ISAAC_PATH/kit/libcarb.so"  # libgomp = aarch64 quirk
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 CUDA_HOME=/usr/local/cuda
```

### 10.5 Per-family run commands

All families use the same launcher `capx/envs/launch.py` with a `--config-path`
from `env_configs/`; only the **venv** (and BEHAVIOR's Isaac env) differ. Pick the
config for the task; `--model`/`--server-url` point at the **peer** LLM (§10.1).

**Oracle smoke-test (no LLM) — verify the substrate first.** Oracle configs run
pre-defined code instead of querying a model, so they need **no** `--model`/`--server-url`
and **no** peer. Use one to validate the venv + perception + pyroki wiring before
spending an LLM run:

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path <oracle-config>.yaml \
    --use-oracle-code True            # NO --model / --server-url; oracle needs no LLM
```

**Robosuite — main `.venv` (`[robosuite]` extra):**
```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model <model> --server-url http://<peer-host>:8000/v1/chat/completions
```

> **Robosuite needs its OWN Franka pyroki server** (stock launcher's default URDF =
> panda/Franka) on a **distinct port** per the §10.2 one-per-family rule. Pre-launch a
> Franka pyroki and point this run at its port — do **not** reuse an R1Pro pyroki: it
> has no Franka `/ik` route. (The GPU-venv pyroki launch mechanism is being finalized —
> §10.2.)

**LIBERO-PRO — `.venv-libero` (LIBERO-PRO fork, robosuite 1.4 + contact-graspnet):**
```bash
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model <model> --server-url http://<peer-host>:8000/v1/chat/completions
```

**BEHAVIOR R1Pro — b1k venv + Isaac wiring (§10.4):**
```bash
source capx/third_party/b1k/.venv/bin/activate
# ... export the §10.4 Isaac env (ISAAC_PATH, shims, LD_PRELOAD, EULA, HEADLESS) ...
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model <model> --server-url http://<peer-host>:8000/v1/chat/completions
```

> **BEHAVIOR GPU note:** Isaac selects its GPU via `OMNIGIBSON_GPU_ID` (not
> `CUDA_VISIBLE_DEVICES`). On this single-GPU head everything shares GPU 0; watch
> unified-memory headroom (Isaac + GPU pyroki + perception all on one GPU).

---

## 11. Known gaps / TODO

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
2. **Isaac path — parameterized (no longer a code TODO).** The `ISAAC_PATH`
   parameterization landed in [`KE7/b1k` PR #3](https://github.com/KE7/b1k/pull/3)
   (`uv_install.sh` resolves `$ISAAC_PATH` from the env, else autodetects it
   repo-relative — §9). Only the path **value** is intrinsically per-machine; set it
   per box. Nothing is hard-coded.
3. **`uv_install.sh` aarch64 port — committed, merge pending.** The edits are
   **committed** in `KE7/b1k` PR #3 (branch `fix/aarch64-uv-install-isaac51-source-build`,
   head `61d8af9e`; OPEN). The remaining step is the **submodule repoint** — cap-x's
   b1k submodule still pins `qingh097/b1k@272ec5ca`; tracked by cap-x PR #6
   (`build/submodule-repoint-ke7-forks`). The curobo fix is likewise done and
   committed — see §9 commits `93695c82e` + `c5182e88f`.
   `picking_up_trash` motion-gen E2E remains **unrun** (radio is the validated task; §9).
4. **Durable lock (eventual reproducibility PR).** The clean long-term fix is to
   add `aarch64` to `pyproject.toml [tool.uv] environments` and move the aarch64
   pins (open3d, decord gate) into `override-dependencies`, then `uv lock`. That
   touches shared `pyproject.toml`/`uv.lock` and is out of scope for this docs
   branch. The overrides file + scripts here are the interim reproducible path.
5. **pyroki bind address** — the server binds `127.0.0.1` and `main()` ignores
   `--host/--port`; needs argv wiring for non-default ports / remote serving.
6. **decord & open3d-on-py3.12** require source builds (§8) if ever needed.
