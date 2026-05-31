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
| BEHAVIOR | `capx/third_party/b1k/.venv` | OmniGibson + Isaac Sim 5.1 | manual, see §9 (in progress) |

robosuite (1.5) and LIBERO (robosuite 1.4) declare a conflict, so they need
separate venvs. pyroki runs GPU JAX (numpy 2.x) which is incompatible with the
benchmark venvs' numpy 1.26.4, so it is isolated and reached over HTTP.

> † `scripts/setup_pyroki_venv.sh` and the launcher's per-venv routing currently
> live on branch **`feat/pyroki-gpu-venv`** (not yet merged to `main`/this docs
> branch). Until that lands, get the pyroki script from that branch.

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
  torch torchvision torchaudio
```

Verified result on this box: **`torch 2.12.0+cu130`, `torch.cuda.is_available() == True`,
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
git -C capx/third_party/curobo apply patches/curobo-lerp.patch
# or:  ( cd capx/third_party/curobo && patch -p1 < ../../../patches/curobo-lerp.patch )
```

This is currently a **working-tree edit to the curobo submodule** (not committed
upstream). The patch file is the reproducible capture; consider upstreaming.

---

## 7. pyroki GPU service (`.venv-pyroki`)

> **Note:** the `setup_pyroki_venv.sh` script and the `launch_servers.py`
> per-venv routing live on branch **`feat/pyroki-gpu-venv`** and are not yet on
> `main` (this docs branch). Run the script from that branch until it merges.

```bash
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

## 9. BEHAVIOR (OmniGibson + Isaac Sim 5.1) — IN PROGRESS

> ⚠️ **Status: partially working; one hard blocker (curobo on CUDA 12.8) is
> still open at the time of writing — see §10.** The b1k submodule work lives on
> branch `feat/omnigibson-3.8.0-isaac5.1` and is owned by a separate worker;
> this section documents the wiring, it does not script it.

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

**Local CUDA 12.8 curobo build (the open blocker):** Isaac's bundled torch is
`2.7.0+cu128`, and curobo must build against that exact torch, but the only
system toolkit is CUDA 13.0. The fix in progress is a **local, non-permanent**
CUDA 12.8 toolkit (does **not** touch `/usr/local/cuda`), with
`CUDA_HOME=<local-12.8>`, `TORCH_CUDA_ARCH_LIST=12.1+PTX`, `--no-build-isolation`
under the wired env. Until curobo builds, the full `picking_up_trash` motion-gen
E2E is deferred (the modified holonomic-base helpers were verified directly on a
live R1Pro; 34 unit tests pass).

---

## 10. Known gaps / TODO

1. **BEHAVIOR curobo (CUDA 12.8) unresolved.** Needs a local CUDA 12.8 toolkit to
   match Isaac's cu128 torch; build in progress. Full motion-gen E2E blocked until
   it lands.
2. **Machine-specific Isaac path** (`/home/batman/Documents/open-source/isaacsim/...`)
   is hard-coded to this box — parameterize via `ISAAC_PATH` and confirm per machine.
3. **`uv_install.sh` edits uncommitted** on the b1k branch; committing them (and a
   BEHAVIOR setup script) is pending the curobo fix + coordination with the b1k owner.
4. **Durable lock (eventual reproducibility PR).** The clean long-term fix is to
   add `aarch64` to `pyproject.toml [tool.uv] environments` and move the aarch64
   pins (open3d, decord gate) into `override-dependencies`, then `uv lock`. That
   touches shared `pyproject.toml`/`uv.lock` and is out of scope for this docs
   branch. The overrides file + scripts here are the interim reproducible path.
5. **pyroki bind address** — the server binds `127.0.0.1` and `main()` ignores
   `--host/--port`; needs argv wiring for non-default ports / remote serving.
6. **decord & open3d-on-py3.12** require source builds (§8) if ever needed.
