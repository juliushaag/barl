"""The value range a categorical head spreads its bins over.

- `FixedSupport`    `[v_min, v_max]` from config, else the env's declared return
  range in reward-scaled units; computed once and kept as buffers.
- `LearnedSupport`  DySEL (Learning the Supports for Categorical Critic in RL,
  arXiv:2607.01880): the endpoints are trained. The support minimises
  `alpha * max(|v_min|, |v_max|)` subject to the expected mass the TD target
  leaks outside it staying below `eps`, enforced by a Lagrange multiplier trained
  by gradient ascent. The head's cross-entropy is scaled by 1 / `alpha`, and the
  Gaussian the leak is measured under is floored at `sigma_min` so shrinking it
  cannot fake coverage. Trained under its own optimizer keys `support` and
  `support_dual`.

The bins stay `num_bins` equal-width and index-fixed while the range moves; the
head owns their layout (`layout`), the support only where they sit.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer, Part
from barl_jax.config.serialization import variant_registry
from barl_jax.optimization import adam

SUPPORTS, register_support = variant_registry()


def value_range(env_stats, v_min, v_max, reward_scale, fallback=None) -> tuple[float, float]:
  """Configured bounds win; otherwise the env's return range in reward-scaled units."""
  bounds = []
  for configured, stat in ((v_min, "return_min"), (v_max, "return_max")):
    if configured is not None:
      bounds.append(float(configured))
    elif env_stats.get(stat) is not None:
      bounds.append(float(env_stats[stat]) * reward_scale)
    elif fallback is not None:
      bounds.append(fallback[len(bounds)])
    else:
      raise KeyError(f"{stat} is missing from env_stats and not configured")
  return tuple(bounds)


@dataclass(frozen=True)
class SupportConfig:
  variants: ClassVar[dict[str, type]] = SUPPORTS

  class_name: str = ""

  def problems(self) -> list[str]:
    return []


@register_support
@dataclass(frozen=True)
class FixedSupportConfig(SupportConfig):
  class_name: str = "FixedSupport"
  v_min: float | None = None
  v_max: float | None = None

  def problems(self) -> list[str]:
    if self.v_min is not None and self.v_max is not None and not self.v_min < self.v_max:
      return ["v_min must be < v_max"]
    return []

  def build(self, head, env_stats, reward_scale) -> "FixedSupport":
    return FixedSupport(self, head, env_stats, reward_scale)


@register_support
@dataclass(frozen=True)
class LearnedSupportConfig(SupportConfig):
  class_name: str = "LearnedSupport"
  init_min: float = -10.0
  init_max: float = 10.0
  alpha: float = 0.3
  eps: float = 0.005
  sigma_min: float = 0.3
  lr: float = 3.0e-3
  dual_lr: float = 1.0e-3
  dual_init: float = 1.0

  def problems(self) -> list[str]:
    problems = []
    if not self.init_min < self.init_max:
      problems.append("init_min must be < init_max")
    for name in ("alpha", "sigma_min", "lr", "dual_lr"):
      if not getattr(self, name) > 0:
        problems.append(f"{name}={getattr(self, name)!r}: must be > 0")
    if not 0 < self.eps < 1:
      problems.append(f"eps={self.eps!r}: must be in (0, 1)")
    if self.dual_init < 0:
      problems.append(f"dual_init={self.dual_init!r}: must be >= 0")
    return problems

  def build(self, head, env_stats, reward_scale) -> "LearnedSupport":
    return LearnedSupport(self)


class Support(Part):
  learned = False

  def bounds(self):
    """`(lo, hi)`, a constant to every loss."""
    raise NotImplementedError

  def geometry(self, head):
    """The head's current `(edges or None, centers, sigma)`."""
    raise NotImplementedError

  def grid(self, lo, hi, n):
    """`n` evenly spaced values from `lo` to `hi`."""
    return jnp.linspace(lo, hi, n, dtype=jnp.float32)

  def floor_sigma(self, sigma):
    return sigma

  def scale_loss(self, loss):
    return loss

  def losses(self, head, chunk_return, discount, bootstrap, entropy_bonus) -> dict:
    return {}


class FixedSupport(Support):

  def __init__(self, config: FixedSupportConfig, head, env_stats, reward_scale):
    super().__init__(config)
    self.v_min, self.v_max = value_range(env_stats, config.v_min, config.v_max, reward_scale)
    edges, centers, sigma = head.layout(self, self.v_min, self.v_max)
    self.centers = Buffer(centers)
    self.edges = None if edges is None else Buffer(edges)
    self.sigma = float(sigma) if head.reports_sigma else None

  def bounds(self):
    return self.v_min, self.v_max

  def geometry(self, head):
    return None if self.edges is None else self.edges[...], self.centers[...], self.sigma


class LearnedSupport(Support):
  """The two trained endpoints, sorted on read and kept a minimum width apart so
  the interval is valid whatever the optimizer does to them, and the dual."""

  learned = True
  MIN_WIDTH = 1.0e-2

  def __init__(self, config: LearnedSupportConfig):
    super().__init__(config)
    self.endpoints = nnx.Param(jnp.asarray([config.init_min, config.init_max], jnp.float32))
    self.dual = nnx.Param(jnp.asarray(config.dual_init, jnp.float32))

  def trained_bounds(self):
    """The endpoints as trained, differentiable."""
    lo, hi = jnp.sort(self.endpoints[...])
    return lo, jnp.maximum(hi, lo + self.MIN_WIDTH)

  def bounds(self):
    return jax.lax.stop_gradient(self.trained_bounds())

  def geometry(self, head):
    return head.layout(self, *self.bounds())

  def grid(self, lo, hi, n):
    # Spelled out so the grid stays differentiable in the endpoints. It differs
    # from `linspace` in the last bit, which cancellation in the cross-entropy
    # gradient can amplify, so each support keeps its own.
    return lo + (hi - lo) * jnp.arange(n, dtype=jnp.float32) / (n - 1)

  def floor_sigma(self, sigma):
    return jnp.maximum(sigma, self.config.sigma_min)

  def scale_loss(self, loss):
    return loss / self.config.alpha

  def create_optimizer(self) -> dict:
    return {"support": adam(self.config.lr, 0.9, 1.0, "none"),
            "support_dual": adam(self.config.dual_lr, 0.9, 1.0, "none")}

  def param_labels(self, subpath):
    return {"endpoints": "support", "dual": "support_dual"}.get(str(subpath[0])) if subpath else None

  def losses(self, head, chunk_return, discount, bootstrap, entropy_bonus) -> dict:
    """The support's Lagrangian: it descends `alpha * max(|v_min|, |v_max|) +
    lambda * (leak - eps)`; lambda ascends `lambda * (leak - eps)`, which a single
    backward pass over the summed losses gets from the negated, stop-gradiented
    copy below. `chunk_return` is reward-scaled."""
    lo, hi = self.trained_bounds()
    _, centers, sigma = head.layout(self, lo, hi)
    leak = head.leak(lo, hi, sigma, jax.lax.stop_gradient(centers), chunk_return, discount,
                     jax.lax.stop_gradient(bootstrap), entropy_bonus)
    violation = leak - self.config.eps
    dual = self.dual[...]

    self.metrics.scalar("min", lo)
    self.metrics.scalar("max", hi)
    self.metrics.scalar("sigma", sigma)
    self.metrics.scalar("leak", leak)
    self.metrics.scalar("multiplier", dual)   # not "dual": that name is the parameter
    width = self.config.alpha * jnp.maximum(jnp.abs(lo), jnp.abs(hi))
    return {"support": width + jax.lax.stop_gradient(dual) * violation,
            "support_dual": -dual * jax.lax.stop_gradient(violation)}

  def after_step(self):
    super().after_step()
    self.dual[...] = jnp.maximum(self.dual[...], 0.0)


def gaussian_mass_outside(lo, hi, y, sigma):
  """Mass of N(y, sigma^2) outside `[lo, hi]`, elementwise."""
  scale = sigma * math.sqrt(2.0)
  return 0.5 * (jax.scipy.special.erfc((hi - y) / scale) + jax.scipy.special.erfc((y - lo) / scale))
