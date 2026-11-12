# Configuration

The training CLI requires an accelerator. Set `BARL_ALLOW_CPU=1` for intentional
CPU smoke tests. Rendering uses EGL by default in the training CLI.

Every run config is preflight-checked (`barl_jax.config.run.validate_run` and the
typed model config's own checks) before an environment, a
policy, or a JAX device buffer is allocated -- an invalid `total_env_steps`,
`checkpoint_mode`, `lr_decay`, etc. fails immediately with the offending key,
value and violated rule, not partway through setup.

`logger:` selects the reporting sink -- `wandb` (default), `local` (the same
live terminal panel, with metrics/summary written as strict JSON under the
run directory instead of shipped to wandb), or `none` (step bookkeeping only,
for a training smoke test with no reporting backend at all). Only `wandb`
imports the `wandb` package. A run's `objective.json` (`trainer._write_objective`) is strict JSON too: a seed with no
completed evaluation is `null` there, not a non-finite float, with
`objective_reason` saying why.

Algorithm defaults live in typed dataclasses beside their implementations
(`StochasticActorConfig`, `C51CriticConfig`, ... with `config_type`, the
single source of truth for that component -- `config/schemas.py` stays
generic model/components scaffolding and never redeclares them); policy YAML
files are no longer required. MLP settings are grouped under one nested `mlp`
dataclass on actors and critics, read back as
`self.config.mlp.activation`/`self.config.mlp.norm`/... There is no capacity-tier
selector; set width/depth directly. For example, change an activation with
`--set model.components.actor.mlp.activation=silu`, or widen a critic with
`--set model.components.critic.latent_dim=1024`.

Hidden linear layers initialize for the activation that follows them: He-style
fan-in scaling for ReLU/ReLU6/leaky ReLU, Xavier with gain 5/3 for tanh, and
fan-in scaling with approximate unit-Gaussian second-moment gains for
SiLU/GELU/Mish. Bare linear projections retain LeCun normal initialization.
All linear biases start at zero. MLP, SimBa, and AdaLNSimBa output heads use
`mlp.head_gain=0.01` by default; for example,
`--set model.components.actor.mlp.head_gain=0.1` changes that kernel gain.
Fourier time embeddings use GELU initialization for their hidden projection
and an unscaled linear output projection. AMDP/DIME still override their
field heads to exactly zero. XQC's unit-norm weight projection overrides
kernel gains, including the small head gain.

## Evaluation RNG

Each `evaluate()` call owns an `eval_seed` (`barl_jax.seeding.derive_seed(base_seed,
"eval", eval_count)`, logged at eval start) independent of every training RNG
stream (policy, collector, replay) -- evaluation never perturbs a training
trajectory, and running the same config from the same seed reproduces the same
eval episodes at the same iteration. `run_episodes()` derives both the
policy's action key and every wave's `env.reset(seed=...)` from that one
seed; a later wave (or a mid-wave slot refill) never replays an earlier one's
episode. `eval_count` is not restored on resume, consistent with resume's
existing "environment episodes restart" semantics.
