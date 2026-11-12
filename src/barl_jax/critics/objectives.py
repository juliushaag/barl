"""Critic training objectives on top of the plain TD regression.

- `TD`                  the Bellman regression alone.
- `NoiseConditionedTD`  NCLQL: the network is also conditioned on a noise level,
  and noisy copies of the batch actions are regressed onto the clean
  prediction (`smoothing_coef`). The sigma ladder is shared with the Langevin
  sampler, which reads it as `critic.objective.sigmas`.
- `CQL`                 CQL(H) (Kumar et al., arXiv:2006.04779) as an online
  regularizer on top of TD: per member, `alpha * (T * logsumexp_a(Q(s, a) / T) -
  Q(s, a_data))` over `n_actions` uniform, `n_actions` current-state and
  `n_actions` next-state policy actions, importance weighted by their densities.
  Q is the head's expected value, so it applies to every head; the penalty is
  averaged over members and batch. The policy draws its samples
  (`RLPolicy.policy_samples`), as `policy_samples` asks.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from barl_jax.components import Buffer, Part
from barl_jax.config.serialization import variant_registry
from barl_jax.networks.schedule import Schedule, ScheduleConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig

OBJECTIVES, register_objective = variant_registry()


@dataclass(frozen=True)
class ObjectiveConfig:
  variants: ClassVar[dict[str, type]] = OBJECTIVES
  # What `Policy.slot_problem` matches against an actor's `critic_objective`.
  learning_objective: ClassVar[str] = "td"

  class_name: str = ""

  def conditioning(self) -> TimeEmbeddingConfig | None:
    """The level encoder the critic network is conditioned on, if any."""
    return None

  def problems(self) -> list[str]:
    return []


@register_objective
@dataclass(frozen=True)
class TDConfig(ObjectiveConfig):
  class_name: str = "TD"

  def build(self, env_stats) -> "TD":
    return TD(self)


@register_objective
@dataclass(frozen=True)
class NoiseConditionedTDConfig(ObjectiveConfig):
  learning_objective: ClassVar[str] = "noise_td"

  class_name: str = "NoiseConditionedTD"
  # One level per entry of the Langevin actor's `steps_per_level`.
  schedule: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(steps=3))
  time: TimeEmbeddingConfig = field(default_factory=FourierTimeConfig)
  smoothing_coef: float = 1.0

  def conditioning(self) -> TimeEmbeddingConfig:
    return self.time

  def problems(self) -> list[str]:
    from barl_jax.registration import get_component
    try:
      is_schedule = issubclass(get_component(self.schedule.class_name), Schedule)
    except KeyError:
      is_schedule = False
    return [] if is_schedule else [f"schedule.class={self.schedule.class_name!r}: unknown schedule"]

  def build(self, env_stats) -> "NoiseConditionedTD":
    return NoiseConditionedTD(env_stats, self)


@register_objective
@dataclass(frozen=True)
class CQLConfig(ObjectiveConfig):
  class_name: str = "CQL"
  alpha: float = 1.0
  n_actions: int = 10
  temperature: float = 1.0

  def problems(self) -> list[str]:
    problems = []
    if self.n_actions < 1:
      problems.append(f"n_actions={self.n_actions!r}: must be >= 1")
    if not self.temperature > 0:
      problems.append(f"temperature={self.temperature!r}: must be > 0")
    if self.alpha < 0:
      problems.append(f"alpha={self.alpha!r}: must be >= 0")
    return problems

  def build(self, env_stats) -> "CQL":
    return CQL(env_stats, self)


class Objective(Part):
  supports_bon = True
  # Policy actions per state the loss needs (`td_loss(policy_samples=...)`); 0 for none.
  policy_samples = 0

  def level(self, s, level):
    """Noise level per row for a forward; None when the network is unconditioned."""
    return None

  def extra_items(self, obs, action, key) -> list:
    """Further `(s, a, level)` batches the live network is evaluated on."""
    return []

  def loss(self, critic, td_loss, preds, **batch):
    """The critic loss given the TD loss on `preds[0]` and predictions for the
    extra items in `preds[1:]`. `batch`: `obs`, `next_obs`, `key` and
    `policy_samples` of the TD step."""
    return td_loss


class TD(Objective):
  pass


class NoiseConditionedTD(Objective):
  """The last level is the clean one; noisy training levels are drawn from the
  ones before it. Pricing with a noise-conditioned critic is not a plain Q, so
  best-of-N scoring is unavailable."""

  supports_bon = False

  def __init__(self, env_stats, config: NoiseConditionedTDConfig):
    super().__init__(config)
    from barl_jax.registration import get_component
    schedule_cls = get_component(config.schedule.class_name)
    if not issubclass(schedule_cls, Schedule):
      raise TypeError("objective.schedule must name a Schedule")
    self.sigmas = Buffer(schedule_cls(env_stats, config.schedule).ladder())
    if self.sigmas.shape[0] < 2:
      raise ValueError("noise-conditioned TD requires at least two noise levels")

  @property
  def clean_level(self) -> int:
    return self.sigmas.shape[0] - 1

  def level(self, s, level):
    level = self.clean_level if level is None else level
    return jnp.broadcast_to(jnp.asarray(level, jnp.int32), (s.shape[0],))

  def perturb(self, action, key):
    level_key, noise_key = jax.random.split(key)
    levels = jax.random.randint(level_key, (action.shape[0],), 0, self.clean_level)
    noise = jax.random.normal(noise_key, action.shape, action.dtype)
    return action + self.sigmas[...][levels][:, None] * noise, levels

  def extra_items(self, obs, action, key) -> list:
    if key is None:
      raise ValueError("noise-conditioned TD requires an explicit random key")
    noisy, levels = self.perturb(action, key)
    return [(obs, noisy, levels)]

  def loss(self, critic, td_loss, preds, **batch):
    smoothing = critic.regression_loss(preds[1], jax.lax.stop_gradient(critic.soft_target(preds[0])))
    self.metrics.scalar("clean_td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing)
    return td_loss + self.config.smoothing_coef * smoothing


class CQL(Objective):
  """Keeps the action bounds the uniform proposals are drawn from."""

  def __init__(self, env_stats, config: CQLConfig):
    super().__init__(config)
    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    ones = np.ones(env_stats["act_dim"], np.float32)
    self.act_bounds = Buffer(jnp.asarray(np.stack([-ones if low is None else low,
                                                   ones if high is None else high]), jnp.float32))

  @property
  def policy_samples(self) -> int:
    return self.config.n_actions

  def penalty(self, critic, obs, next_obs, data_expectation, samples, key):
    """`data_expectation`: (N, B, 1) member expectations at the dataset actions;
    `samples`: `(curr, curr_logp, next, next_logp)` policy actions (n, B, A) and
    their log-densities (n, B)."""
    cfg = self.config
    curr, curr_logp, nxt, nxt_logp = samples
    n, b = curr.shape[:2]
    low, high = self.act_bounds[...]
    rand = jax.random.uniform(key, curr.shape, jnp.float32, low, high)
    rand_logp = jnp.full((n, b), -jnp.log(high - low).sum())

    states = jnp.concatenate([jnp.broadcast_to(obs, (n,) + obs.shape)] * 2
                             + [jnp.broadcast_to(next_obs, (n,) + next_obs.shape)])
    actions = jnp.concatenate([rand, curr, nxt])
    logp = jnp.concatenate([rand_logp, curr_logp, nxt_logp])
    flat = lambda x: x.reshape((3 * n * b,) + x.shape[2:])
    q = critic.expectation_members(critic.raw(flat(states), flat(actions)))
    q = q.reshape(q.shape[0], 3 * n, b)

    t = cfg.temperature
    lse = t * jax.nn.logsumexp((q - logp[None]) / t, axis=1)
    gap = lse - data_expectation[..., 0]
    self.metrics.scalar("gap", gap.mean())
    self.metrics.scalar("q_sampled", q.mean())
    return cfg.alpha * gap.mean()

  def loss(self, critic, td_loss, preds, *, obs=None, next_obs=None, key=None,
           policy_samples=None, **batch):
    if key is None or policy_samples is None:
      raise ValueError("CQL requires an explicit random key and policy_samples")
    penalty = self.penalty(critic, obs, next_obs, critic.expectation_members(preds[0]),
                           policy_samples, key)
    self.metrics.scalar("td_loss", td_loss)
    self.metrics.scalar("penalty", penalty)
    return td_loss + penalty
