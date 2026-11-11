<p align="center">
  <img src="docs/assets/banner.png" alt="BARL — Best Adaptations Reinforcement Learning" width="100%">
</p>

<p align="center">
  <a href="https://github.com/juliushaag/barl/actions/workflows/ci.yml"><img src="https://github.com/juliushaag/barl/actions/workflows/ci.yml/badge.svg?branch=master" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python ≥ 3.11">
  <img src="https://img.shields.io/badge/JAX-Flax%20NNX-orange" alt="JAX / Flax NNX">
  <img src="https://img.shields.io/badge/lint-ruff-261230" alt="ruff">
</p>

**BARL** is a JAX / Flax-NNX research framework for online, off-policy actor-critic
reinforcement learning. Every algorithm is a combination of interchangeable
components (an actor, a critic, an optional entropy regulator) on one shared
training loop, so swapping a diffusion actor for a Gaussian one, or a C51 critic for a
flow-matching one, is a one-line config change.

- **Seven algorithms, one training step.** SAC, TD3, AMDP, DIME, SDAC, SDAC (paper
  version) and NCLQL share the same collection, replay, evaluation and checkpointing code.
- **Swappable critics.** Scalar, ensemble, HL-Gauss, C51 and flow-matching (floq) critics.
  They can be combined with polyak or CrossQ-style targets, XQC weight normalization and
  noise-conditioned TD.
- **Multi-seed by default.** Seeds are `vmap`ped within a single run on one device.
- **Inference-time action selection.** Best-of-N, top-k, mean, median and perturbation
  executors, all independent of the policy.
- **Simulators.** MuJoCo (Gymnasium), DeepMind Control, HumanoidBench, ManiSkill and MyoSuite.
- **Sweeps anywhere.** One CLI expands grids and runs them locally, in tmux, or on Slurm.
- **Full resume.** Optimizer state, replay buffer, RNGs and counters are restored on resume.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Algorithms](#algorithms)
- [Components](#components)
- [Running experiments](#running-experiments)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Development](#development)

## Installation

BARL uses [uv](https://docs.astral.sh/uv/). The full environment includes CUDA JAX and
every simulator family. HumanoidBench is installed from a git submodule.

```bash
git clone --recursive https://github.com/juliushaag/barl.git
cd barl
uv sync
```

Cloned without `--recursive`? Run `git submodule update --init` before `uv sync`.

## Quickstart

A complete CPU-only round trip: no GPU, W&B account or simulator download needed. It
trains on Gymnasium's `Pendulum-v1`, saves and reloads a checkpoint, and checks that a
resumed run is bit-identical to the original.

```bash
BARL_ALLOW_CPU=1 uv run python -m examples.cpu_quickstart
```

Train SAC on HalfCheetah with three seeds:

```bash
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --seeds 3 --executor local
```

Or use BARL as a library:

```python
from barl_jax.config.run import RunConfig
from barl_jax.policy import Policy

run = RunConfig.from_dict({
  "model": "SAC", "env": "HalfCheetah",
  "model_overrides": {"params": {"policy_delay": 2}},
})
policy = Policy.make("SAC", env, run.model_config, env_stats=env_stats)
```

## Algorithms

| Model       | Actor                                    | Entropy tuning | Reference |
|-------------|------------------------------------------|:--------------:|-----------|
| `SAC`       | Tanh-squashed Gaussian                   | ✓ | Haarnoja et al., 2018 |
| `TD3`       | Deterministic + target smoothing         |   | Fujimoto et al., 2018 |
| `AMDP`      | Controlled-SDE diffusion, adjoint matching | ✓ | [arXiv:2409.08861](https://arxiv.org/abs/2409.08861) |
| `DIME`      | Diffusion sampler with a learned SDE      | ✓ | Celik et al., 2025 |
| `SDAC`      | DDPM, reverse-sampling score matching     |   | Ma et al., 2025 |
| `SDACPaper` | DDPM, as released with the paper          | ✓ | Ma et al., 2025 |
| `NCLQL`     | Annealed Langevin sampling on the critic  |   | [docs/nclql.md](docs/nclql.md) |

Each model's defaults are typed dataclasses next to its implementation. Override any
field with `--set`:

```bash
uv run python scripts/submit.py --algos SAC --envs HalfCheetah \
  --set model.components.actor.mlp.activation=silu \
  --set model.components.critic.latent_dim=1024
```

## Components

| Slot | Built-in options |
|------|------------------|
| **Critic** | `DirectCritic`, `EnsembleCritic`, `HLGaussCritic`, `C51Critic`, `FloqCritic` ([docs](docs/floq.md)) |
| **Critic presets** | `target_ensemble`, `xqc`, `nclql` |
| **Networks** | `MLP`, `SimBa`, `AdaLNSimBa`; norms: `none`, `layer`, `batch`, `batchrenorm` |
| **Squash** | `TanhSquash`, `ErfSquash` |
| **Entropy** | `AutotuneRegulator` (per-chunk or per-action α) |
| **Executors** | `DirectExecutor`, `MeanExecutor`, `MedianExecutor`, `AxisShuffleExecutor`, `BestOfNExecutor`, `TopKExecutor`, `PerturbExecutor` ([docs](docs/executors.md)) |

You can also select your own classes with a `module:Class` path or through an installed
plugin entry point. See [docs/extending.md](docs/extending.md).

## Running experiments

`scripts/submit.py` expands a grid and hands each run to an executor (`local`, `tmux` or
`slurm`). It never trains anything itself; it calls `scripts/train.py` and
`scripts/evaluate.py`, just as you would run them by hand.

```bash
# Preview a grid without running it
uv run python scripts/submit.py --algos SAC NCLQL --envs HalfCheetah --seeds 3 --dry-run

# Run a committed experiment grid
uv run python scripts/submit.py --grid experiments/sac_floq_critic.yaml --seeds 2 --executor tmux

# Replay one saved run
uv run python scripts/submit.py --config sweeps/<submission>/<run>/config.yaml

# Evaluate a sweep under several action-selection strategies
uv run python scripts/evaluate.py sweeps/<submission> --mode DirectExecutor MeanExecutor --json
```

| Script | Purpose |
|--------|---------|
| `scripts/submit.py`   | Build grids and dispatch runs (local / tmux / Slurm) |
| `scripts/train.py`    | Train one experiment config |
| `scripts/evaluate.py` | Evaluate a checkpoint, run directory or sweep |
| `scripts/debug.py`    | Package/backend diagnostics and resolved model settings (`--json`) |
| `scripts/egl_probe.py`| Diagnose headless EGL rendering on a compute node |

Training requires an accelerator unless `BARL_ALLOW_CPU=1` is set. The `logger:` key
selects `wandb` (default), `local` (JSON under the run directory) or `none`.

## Project layout

```
src/barl_jax/
  actors/        actor implementations and their per-algorithm policies
  critics/       value heads, targets, ensembles and presets
  networks/      MLP, SimBa, norms, time embeddings, squashes, schedules, entropy regulator
  policies/      the shared actor-critic training template
  executors/     inference-time action selection
  environments/  simulator families (gym, dm_control, humanoid_bench, mani_skill, myosuite)
  config/        typed run and model configuration
  trainer.py     run lifecycle: collection, replay, updates, evaluation, checkpoints
launch/          tmux and Slurm backends used by submit.py
scripts/         command-line entry points
experiments/     committed sweep grids
examples/        CPU quickstart
```

## Documentation

| Topic | |
|-------|-|
| Configuration, validation, initialization, evaluation RNG | [docs/configuration.md](docs/configuration.md) |
| Inference-time action executors | [docs/executors.md](docs/executors.md) |
| Checkpoints and exact resume | [docs/checkpoints.md](docs/checkpoints.md) |
| Architecture, plugins and custom components | [docs/extending.md](docs/extending.md) |
| Flow-matching critic (floq) | [docs/floq.md](docs/floq.md) |
| NCLQL action sampling | [docs/nclql.md](docs/nclql.md) |
| MyoSuite tasks | [docs/myosuite.md](docs/myosuite.md) |

## Development

```bash
uv run pytest tests/ -q      # test suite (pytest-xdist, -n 4 by default)
uv run ruff check .          # lint
uv build --wheel             # build the package
```

CI runs ruff, the CPU test suite, an installed-wheel import check and the CPU quickstart.
It installs the package without its dependencies, plus a CPU-only dependency set, so it
needs no CUDA or simulator downloads.
