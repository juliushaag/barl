# Components and parts

A model is a set of **components** in named slots (`actor`, `critic`, `entropy`).
A component can be split further into **parts**: swappable pieces with their own
typed config, each selected by `class:`. Everything that applies to every
trainable component, such as parameter **constraints**, is configured the same way
on each of them.

```yaml
model_overrides:
  components:
    critic:
      class: QCritic                      # or a preset name: C51Critic, HLGaussCritic, ...
      head:
        class: C51                        # Scalar | C51 | HLGauss
        num_bins: 101
        support: {class: LearnedSupport, alpha: 0.3}   # or {class: FixedSupport, v_min, v_max}
      target: {class: Polyak, tau: 0.005, joint_forward: true}   # or {class: NoTarget}
      objective: {class: TD}              # or {class: NoiseConditionedTD, smoothing_coef: 1.0}
      ensemble: {n_models: 2, value_reduce: mean, target_reduce: min}
      mlp: {class: MLP, norm: batchrenorm}
      constraints: [{class: WeightNorm}]
    actor:
      constraints: [{class: WeightNorm}]  # constraints work on actors too
```

From the command line, address any field by its path:

```bash
--set model.components.critic.head.support.class=FixedSupport
--set model.components.critic.target.class=Polyak
--set 'model.components.actor.constraints=[{class: WeightNorm}]'
```

## The critic's parts

`QCritic` (`critics/base.py`) owns the network and the TD step; the rest is parts.

| Part | Options | Module |
|------|---------|--------|
| `head` | `Scalar` (MSE), `C51` (projected atoms), `HLGauss` (Gaussian-smoothed histogram) | `critics/heads.py` |
| `head.support` | `FixedSupport` (`v_min`/`v_max` or the env's return range), `LearnedSupport` (DySEL) | `critics/supports.py` |
| `target` | `NoTarget` (bootstrap through the frozen live net, CrossQ), `Polyak` (`tau`, `delay`) | `critics/targets.py` |
| `objective` | `TD`, `NoiseConditionedTD` (NCLQL) | `critics/objectives.py` |
| `ensemble` | `n_models`, `value_reduce` / `target_reduce` = `mean` \| `min` | `critics/base.py` |

`DirectCritic`, `EnsembleCritic`, `C51Critic` and `HLGaussCritic` are *presets*:
registered names whose only difference is their default parts. `FloqCritic`
replaces the network and the regression itself. It reuses the target part and
the scalar head, and states what it cannot combine with in its own
`config_problems`.

Named recipes (`target_ensemble`, `xqc`, `nclql`) are partial configs in
`critics/presets.py`, applied with `apply_preset(config, "xqc")`.

## One path for config, state and metrics

A part's config path, its place in the module tree and its metric namespace are
the same thing:

| Config | State | Metrics |
|--------|-------|---------|
| `critic.head.support` | `critic.head.support.endpoints`, `.dual` | `critic/head/support/{min,max,sigma,leak,multiplier}` |
| `critic.target` | `critic.target.net_target`, `.steps` | `critic/target/gap` |
| `critic.objective` | `critic.objective.sigmas` | `critic/objective/{clean_td_loss,smoothing_loss}` |

A part that trains parameters of its own declares its optimizer keys
(`LearnedSupport`: `support`, `support_dual`). The component collects them, along
with parameter ownership and the `after_step` hooks, automatically. These losses
are logged as `critic/loss/support` and `critic/loss/support_dual`.

## Constraints

`constraints` is a field of every trainable component's config. Each one projects
every network trunk inside the component (any module with `hidden_linears` and
`head`: MLP, SimBa, AdaLN-SimBa, ensembles included, target copies excluded):

- after the component is built, before a target network is cloned from it;
- after every optimizer step, before the component's own `after_step`, so a
  Polyak target blends towards the projected weights.

`WeightNorm` (XQC's unit-norm projection) is the built-in. A new constraint is
one registered `ConstraintConfig` subclass with an `apply(trunk)` method, in
`networks/constraints.py`.

## Adding a part

1. Write a config dataclass and register it with the family's decorator
   (`register_head`, `register_support`, `register_target`, `register_objective`).
   Its `class_name` is the name used in YAML, and it gets a `build(...)` method.
2. Write the part as a `Part` subclass. Assign sub-parts as attributes: their
   optimizers, parameter labels and hooks are picked up automatically.
3. If a combination of parts can't work, add a rule to `COMBINATION_RULES` in
   `critics/base.py`, so it is reported before anything is built.
4. The critic test grid in `tests/conftest.py` crosses heads, targets, norms,
   reductions and objectives. Add the new option there.

## Backward compatibility

All of it lives in `src/barl_jax/compat/`, and nothing elsewhere special-cases
an old format:

- `compat/config.py` rewrites component mappings written for an older schema:
  experiment YAML, `--set` overrides and the model spec stored in a checkpoint.
  For example `weight_norm: true` → `constraints: [{class: WeightNorm}]`,
  `learn_support: false` → `head.support.class: FixedSupport`,
  `target.kind: polyak` → `target.class: Polyak`, and
  `noise.enabled: true` → `objective.class: NoiseConditionedTD`.
- `compat/checkpoint.py` holds the checkpoint format versions and the stepwise
  state migrations (renamed parameter paths). Policy checkpoints and resume
  checkpoints both go through it, because optimizer moments sit under the same
  paths as their parameters.

`tests/test_compat.py` loads real checkpoints written by the previous release
(`tests/fixtures/policy_v3/`) and checks that they reproduce the Q-values,
actions and training losses recorded when they were saved. To change a saved
layout, bump the version, add a migration, and add a fixture written by the old
code.
