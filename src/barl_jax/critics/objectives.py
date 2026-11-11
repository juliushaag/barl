"""Critic training objectives on top of the plain TD regression.

- `TD`                  the Bellman regression alone.
- `NoiseConditionedTD`  NCLQL: the network is also conditioned on a noise level,
  and noisy copies of the batch actions are regressed onto the clean
  prediction (`smoothing_coef`). The sigma ladder is shared with the Langevin
  sampler, which reads it as `critic.objective.sigmas`.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp

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
  schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
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


class Objective(Part):
  supports_bon = True

  def level(self, s, level):
    """Noise level per row for a forward; None when the network is unconditioned."""
    return None

  def extra_items(self, obs, action, key) -> list:
    """Further `(s, a, level)` batches the live network is evaluated on."""
    return []

  def loss(self, critic, td_loss, preds):
    """The critic loss given the TD loss on `preds[0]` and predictions for the
    extra items in `preds[1:]`."""
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

  def loss(self, critic, td_loss, preds):
    smoothing = critic.regression_loss(preds[1], jax.lax.stop_gradient(critic.soft_target(preds[0])))
    self.metrics.scalar("clean_td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing)
    return td_loss + self.config.smoothing_coef * smoothing
