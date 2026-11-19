# Extending the framework

- `components.py` defines actors, critics, configuration declarations and NNX
  state utilities. Low-level components do not import the policy runtime.
- `policies/` composes components and defines algorithm behavior. Critic losses
  live in `critics/`: `QCritic` owns the network and the TD step and is composed
  of parts -- a value head (`Scalar`, `C51`, `HLGauss`) with its support, a
  bootstrap target and a training objective (see [components.md](components.md)).
  `SLOT_OBJECTIVES` validates compatibility before construction.
- `optimization.py` assembles parameter ownership, update cadence and schedules.
- `collection.py` owns environment state and emits transitions into an
  injected sink. `replay.py` owns replay storage and independent sampling RNGs.
- `evaluation.py` evaluates action providers; `multiseed.py` stacks independent
  policies and optimizers. `trainer.py` coordinates their lifecycle.
- `checkpoints.py` restores policy architectures from saved metadata;
  `training_checkpoint.py` handles optimizer/replay/counter restoration.

Importing `barl_jax` or its config toolkit does not initialize JAX or simulators.
Registry lookups load built-ins on demand. Environment-family imports expose
metadata; simulator dependencies load when `get_env()` runs. External policy,
component and environment classes can be selected with `module:Class` paths.
Async environments pickle their bound factory, preserving instance settings;
the factory and its class must be importable in the worker process.

Three ways to name a Policy/Environment/component implementation, tried in
order: a built-in name; an installed plugin -- an entry point under
`barl_jax.policies`/`barl_jax.environments`/`barl_jax.components` in another
distribution's `pyproject.toml` (`[project.entry-points."barl_jax.policies"]`),
discovered once per process and loaded only when actually selected; or
`module:Class`, unaffected by what plugins are installed. A name shared by a
built-in and a plugin, or by two plugins, is a deterministic error at lookup
time, not a silent precedence rule.

For independent in-process experiments, pass configuration explicitly:

```python
from barl_jax.config.run import RunConfig
from barl_jax.policy import Policy

run = RunConfig.from_dict({
  "model": "SAC", "env": "HalfCheetah",
  "model_overrides": {"params": {"policy_delay": 2}},
})
policy = Policy.make("SAC", env, run.model_config, env_stats=env_stats)
```

`RunConfig.model_config` is the complete typed model (policy params and component
configs). Policies retain it as `policy.config`; it is frozen, so later config
changes cannot affect their components. Import contracts, checkpoint functions and inference utilities from
their owning modules. Older checkpoint formats are rejected explicitly, except that
saved configs naming keys this version no longer has (for example the removed
action-chunking params `chunk_size` and `alpha_granularity`) still load: the keys
are dropped with a warning. A checkpoint trained with `chunk_size > 1` cannot be loaded.

Every actor and critic normalizes its own observations through its `input_norm`
part, `RSNorm` (running per-feature mean and std, SimBa-style) by default. The
Policy hands each component the training batch's observations before the losses
run (`Policy.update_observation_stats` -> `component.observe`); inference only
reads the statistics. Its options are `eps` (variance floor) and `clip`
(normalized values are clipped to `+-clip`, default 10, `null` disables it):

```bash
--set model.components.critic.input_norm.clip=5
```

A per-sample norm (LayerNorm/RMSNorm) is not a substitute: it rescales a whole
observation by one number, so a feature of std 60 still drowns out the action.
`Identity` exists only so checkpoints trained without normalization (format <= 4,
`params.normalize_obs: false`) load as they were trained; a new run with it is
rejected at preflight. A new actor or critic gets the part by inheriting
`ActorOptimConfig` / `QCriticConfig` and calls `self.normalize_input(obs)` before
its first network.
