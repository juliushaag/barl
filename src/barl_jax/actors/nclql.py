"""Annealed Langevin action sampling from a noise-conditioned critic."""
import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.components import ActionSampler, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.critics.presets import nclql
from barl_jax.metrics import DEBUG
from barl_jax.networks.base import DTYPES
from barl_jax.policies.rl import PolicyConfig, RLPolicy
from barl_jax.registration import register_component, register_policy


@dataclass(frozen=True)
class LangevinActorConfig:
  class_name: str = "LangevinActor"
  # Substeps at each noise level, coarsest first; an int runs that many at every level.
  steps_per_level: int | list[int] = field(default_factory=lambda: [2, 3, 5])
  step_lr: float = 1.0e-4
  max_step_size: float | None = 1.0e-3
  temperature: float = 500.0
  normalize_score: bool = True
  clip_actions: bool = True
  action_limit: float = 1.0
  init_scale: float = 1.0
  # Precision of the critic copy the chain climbs. The chain is bound by reading
  # the critic's weights, so a "bfloat16" copy (weights stored and computed in
  # bf16; the critic itself is untouched) roughly halves the cost of a large
  # batch. It does nothing for a handful of rows, where the chain is
  # latency-bound, and the copy is rebuilt on every call.
  sample_dtype: str = "float32"

  def problems(self):
    if self.sample_dtype not in DTYPES:
      return [f"sample_dtype={self.sample_dtype!r}: must be one of {sorted(DTYPES)}"]
    steps = [self.steps_per_level] if isinstance(self.steps_per_level, int) else self.steps_per_level
    if not steps or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in steps):
      return ["steps_per_level must be an int >= 1 or a non-empty list of ints >= 1, "
              "one per noise level"]
    if self.max_step_size is not None and not (
        math.isfinite(self.max_step_size) and self.max_step_size > 0):
      return ["max_step_size must be None or finite and > 0"]
    return []


def reduced_precision(module, dtype):
  """A copy of `module` whose parameters are stored in `dtype` and whose layers
  compute in it. Statistics (BatchStat) stay in float32."""
  copy = nnx.clone(module)
  for _, child in nnx.iter_modules(copy):
    if hasattr(child, "dtype"):
      child.dtype = dtype
  nnx.update(copy, jax.tree.map(lambda x: x.astype(dtype), nnx.state(copy, nnx.Param)))
  return copy


@register_component
class LangevinActor(ActionSampler):
  """No policy network or optimizer: climb the live critic's action gradients.

  The critic owns the noise ladder used by both sampling and smoothing loss.
  Score norms are diagnostics, never an entropy estimate.
  """

  config_type = LangevinActorConfig
  critic_objective = "noise_td"

  def __init__(self, env_stats, config: LangevinActorConfig, *, rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))

  def critic_score(self, critic, obs, act, level):
    """grad_a E[Z(s, a, level)], averaged over heads -- `expectation()` and not
    `value()`, since here the gradient field *is* the policy and a pessimism
    term would change the distribution the chain samples from.

    Uses nnx.grad on the frozen critic to avoid trace-level issues when called
    from within a vmapped context (multi-seed rollout)."""
    critic = frozen(critic)

    def q_fn(critic, a):
      return critic.expectation(obs, a, level=level).sum()

    return nnx.grad(q_fn, argnums=1)(critic, act)

  def generate_action(self, obs, *, critic=None, key=None):
    if critic is None:
      raise ValueError("LangevinActor requires a noise-conditioned critic")
    key = self.next_key(key)
    config = self.config
    sigmas = critic.objective.sigmas[...]
    if config.sample_dtype != "float32":
      critic = reduced_precision(critic, DTYPES[config.sample_dtype])
    steps = config.steps_per_level
    steps = (steps,) * int(sigmas.size) if isinstance(steps, int) else tuple(steps)
    if len(steps) != sigmas.size:
      raise ValueError(f"steps_per_level has {len(steps)} entries but the critic's noise "
                       f"ladder has {sigmas.size} levels")
    key, init_key = jax.random.split(key)
    a = config.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.action_dim), jnp.float32) - 1.0)

    # Song & Ermon spacing: `step_lr` is the finest level's step, coarser levels scale
    # by (sigma_i / sigma_last)^2. Cap alpha before computing both drift and noise
    # so coarse updates stay small relative to the action box.
    level_alpha = config.step_lr * (sigmas / sigmas[-1]) ** 2
    if config.max_step_size is not None:
      level_alpha = jnp.minimum(level_alpha, config.max_step_size)

    # One scan over the flattened (level, substep) schedule: the step body is
    # traced once however long the chain is, instead of once per substep.
    levels = np.repeat(np.arange(len(steps)), steps)
    xs = (jnp.asarray(levels, jnp.int32), level_alpha[levels],
          jax.random.split(key, len(levels)))

    def substep(a, x):
      level, alpha, noise_key = x
      g = self.critic_score(critic, obs, a, level)
      score_norm = jnp.linalg.norm(g, axis=-1)
      if config.normalize_score:
        g = g / (score_norm[:, None] + 1e-8)
      drift = 0.5 * alpha * config.temperature * g
      a = a + drift + jnp.sqrt(alpha) * jax.random.normal(noise_key, a.shape, a.dtype)
      clipped = (jnp.abs(a) > config.action_limit).astype(jnp.float32).mean()
      if config.clip_actions:
        a = jnp.clip(a, -config.action_limit, config.action_limit)
      return a, (score_norm.mean(), jnp.linalg.norm(drift, axis=-1).mean(), clipped)

    a, (score_norms, drift_norms, clipped) = jax.lax.scan(substep, a, xs)

    # Per-level diagnostics report the level's last substep.
    last = np.cumsum(steps) - 1
    for i in range(len(steps)):
      self.metrics.scalar(f"chain/step_size_l{i}", level_alpha[i], tier=DEBUG)
      self.metrics.scalar(f"chain/drift_norm_l{i}", drift_norms[last[i]], tier=DEBUG)
      self.metrics.scalar(f"chain/noise_scale_l{i}", jnp.sqrt(level_alpha[i]), tier=DEBUG)
      if config.clip_actions:
        self.metrics.scalar(f"chain/clipped_frac_l{i}", clipped[last[i]], tier=DEBUG)

    self.metrics.scalar("score_norm", score_norms[-1])
    return a, None


@dataclass(frozen=True)
class NCLQLPolicyConfig(PolicyConfig):
  # generate_action's Langevin chain runs through the critic on every grad step
  # (target next-action) and every rollout step, ~3x a plain critic step's cost
  # -- measured on DogRun -- so the default utd is cut from the RLPolicy
  # baseline to keep wall time per env step comparable.
  utd_ratio: float = 0.5


@register_policy
class NCLQL(RLPolicy):
  """Binds LangevinActor into the shared actor-critic template -- the
  actor-policy integration for this algorithm. No entropy dual: the sampler's
  "policy" is the noise-conditioned critic's own gradient field, so this pins
  `RLPolicy` directly rather than `MaxEntRLPolicy`."""

  actor_cls = LangevinActor


@register_model
@dataclass(frozen=True)
class NCLQLModelConfig(ModelConfig):
  policy_class: str = "NCLQL"
  params: NCLQLPolicyConfig = field(default_factory=NCLQLPolicyConfig)
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=LangevinActorConfig(),
    critic=nclql(C51CriticConfig()), entropy=None))
