"""Input normalization: the first thing an actor or critic does to an observation.

Every actor and critic that runs a network on observations owns one of these
parts as `input_norm`, configured per component:

    model.components.actor.input_norm:  {class: RSNorm}
    model.components.critic.input_norm: {class: RSNorm, clip: 10.0}

It has to work *per feature*. A per-sample norm (LayerNorm, RMSNorm) rescales a
whole observation by one number, which leaves a feature of std 60 still 60x
larger than an action in [-1, 1]; on DMC Dog that drowns the action out of a
critic entirely. So the variants here normalize each feature by its own
statistics:

- `RSNorm`    running statistics normalization (as in SimBa): the pooled mean
  and std of every observation the component has trained on, merged from each
  training batch by the owning Policy before the losses run. Normalization
  reads the statistics only, so `(s, a)` and `(s', a')` -- and every network of
  a critic, target and distilled copies included -- see the same mapping.
- `Identity`  no normalization. Exists only so checkpoints trained without
  input normalization load unchanged; a new run with it is rejected at preflight.
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from barl_jax.components import Buffer, Part
from barl_jax.config.serialization import variant_registry

INPUT_NORMS, register_input_norm = variant_registry()


@dataclass(frozen=True)
class InputNormConfig:
  variants: ClassVar[dict[str, type]] = INPUT_NORMS
  # Whether a new run may use it; `Identity` is for old checkpoints only.
  normalizes: ClassVar[bool] = True

  class_name: str = ""

  def problems(self) -> list[str]:
    return []

  def build(self, obs_dim: int) -> "InputNorm":
    raise NotImplementedError


@register_input_norm
@dataclass(frozen=True)
class RSNormConfig(InputNormConfig):
  class_name: str = "RSNorm"
  # Floor on the variance: a constant feature maps to 0, not to 0/0.
  eps: float = 1.0e-8
  # Normalized values are clipped to +-clip; None disables clipping.
  clip: float | None = 10.0

  def problems(self) -> list[str]:
    problems = []
    if not self.eps >= 0:
      problems.append(f"eps={self.eps!r}: must be >= 0")
    if self.clip is not None and not self.clip > 0:
      problems.append(f"clip={self.clip!r}: must be None or > 0")
    return problems

  def build(self, obs_dim: int) -> "RSNorm":
    return RSNorm(self, obs_dim)


@register_input_norm
@dataclass(frozen=True)
class IdentityInputNormConfig(InputNormConfig):
  normalizes: ClassVar[bool] = False

  class_name: str = "Identity"

  def build(self, obs_dim: int) -> "IdentityInputNorm":
    return IdentityInputNorm(self)


class InputNorm(Part):

  def observe(self, obs) -> None:
    """Fold a training batch's observations into the statistics, if any."""

  def __call__(self, obs):
    raise NotImplementedError


class RSNorm(InputNorm):
  """Pooled running mean and std of the observations (exact across unequal
  batches). Identity until the first batch arrives."""

  def __init__(self, config: RSNormConfig, obs_dim: int):
    super().__init__(config)
    self.mean = Buffer(jnp.zeros((obs_dim,), jnp.float32))
    self.std = Buffer(jnp.ones((obs_dim,), jnp.float32))
    self.count = Buffer(jnp.zeros((), jnp.float32))

  def observe(self, obs) -> None:
    obs = jax.lax.stop_gradient(obs.astype(jnp.float32)).reshape(-1, self.mean.shape[-1])
    count = jnp.asarray(obs.shape[0], jnp.float32)
    total = self.count[...] + count
    delta = obs.mean(axis=0) - self.mean[...]
    pooled = (self.std[...] ** 2 * self.count[...] + obs.var(axis=0) * count
              + delta ** 2 * self.count[...] * count / total)
    self.mean[...] = self.mean[...] + delta * count / total
    self.std[...] = jnp.sqrt(jnp.maximum(pooled / total, 0.0))
    self.count[...] = total

  def __call__(self, obs):
    scaled = (obs - self.mean[...]) / jnp.sqrt(self.std[...] ** 2 + self.config.eps)
    if self.config.clip is not None:
      scaled = jnp.clip(scaled, -self.config.clip, self.config.clip)
    return scaled.astype(obs.dtype)


class IdentityInputNorm(InputNorm):

  def __call__(self, obs):
    return obs
