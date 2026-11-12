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

Observation normalization is off by default. With `model.params.normalize_obs=true`,
running moments are updated from each training batch before the objective runs;
inference reads those moments without updating them.
