"""Value heads: what the critic network's outputs represent and how they are fit.

Every head sees raw network outputs shaped `(n_models, B, out_dim)`.

- `Scalar`   one value per member, regressed with MSE.
- `C51`      a softmax over `num_bins` atoms spanning the support; the bootstrap
  distribution is projected onto them (Bellemare et al., 2017).
- `HLGauss`  a softmax over `num_bins` equal bins; the scalar TD target is
  smoothed into a histogram with a Gaussian `sigma_to_bin_ratio` bins wide
  (Imani & White, 2018).

The categorical heads place their bins on a `support` part (fixed or learned)
and fit them with cross-entropy plus an optional `entropy_coef` regularizer on
the predicted distribution (loss += coef * sum p log p).
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp

from barl_jax.components import Part
from barl_jax.config.serialization import variant_registry

from .supports import LearnedSupportConfig, SupportConfig, gaussian_mass_outside

HEADS, register_head = variant_registry()


def scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus=0.0):
  return chunk_return + discount * (bootstrap + entropy_bonus)


def categorical_projection(next_probs, atoms, chunk_return, discount, v_min, v_max,
                           entropy_bonus=0.0):
  """Project `next_probs` over the shifted support `chunk_return + discount *
  (atoms + entropy_bonus)` back onto `atoms` by linear interpolation. Linear
  in `next_probs`."""
  num_atoms = atoms.shape[0]
  delta_z = (v_max - v_min) / (num_atoms - 1)
  tz = jnp.clip(chunk_return.reshape(-1, 1) + discount * (atoms.reshape(1, -1) + entropy_bonus),
                v_min, v_max)

  b = (tz - v_min) / delta_z
  l = jnp.clip(jnp.floor(b), 0, num_atoms - 1).astype(jnp.int32)
  u = jnp.clip(jnp.ceil(b), 0, num_atoms - 1).astype(jnp.int32)

  on_atom = (l == u)
  m_l = jnp.where(on_atom, next_probs, next_probs * (u.astype(b.dtype) - b))
  m_u = jnp.where(on_atom, 0.0, next_probs * (b - l.astype(b.dtype)))

  batch_idx = jnp.arange(next_probs.shape[-2])[:, None]
  proj = jnp.zeros_like(next_probs)
  proj = proj.at[..., batch_idx, l].add(m_l)
  return proj.at[..., batch_idx, u].add(m_u)


def to_histogram(value, edges, sigma):
  """HL-Gauss: the mass of N(value, sigma^2) in each bin, renormalized to the
  clipped range."""
  value = jnp.clip(value.reshape(-1, 1), edges[0], edges[-1])
  cdf = 0.5 * (1.0 + jax.scipy.special.erf((edges - value) / (sigma * math.sqrt(2.0))))
  mass = jnp.maximum(cdf[:, 1:] - cdf[:, :-1], 0.0)
  return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)


# ---- configs -----------------------------------------------------------------


@dataclass(frozen=True)
class HeadConfig:
  variants: ClassVar[dict[str, type]] = HEADS

  class_name: str = ""

  def out_dim(self) -> int:
    raise NotImplementedError

  def problems(self) -> list[str]:
    return []


@register_head
@dataclass(frozen=True)
class ScalarHeadConfig(HeadConfig):
  class_name: str = "Scalar"

  def out_dim(self) -> int:
    return 1

  def build(self, env_stats, reward_scale) -> "ScalarHead":
    return ScalarHead(self)


@dataclass(frozen=True)
class CategoricalHeadConfig(HeadConfig):
  num_bins: int = 100
  entropy_coef: float = 0.005
  sigma_to_bin_ratio: float = 0.75
  support: SupportConfig = field(default_factory=LearnedSupportConfig)

  def out_dim(self) -> int:
    return self.num_bins

  def problems(self) -> list[str]:
    problems = []
    if self.num_bins < 2:
      problems.append(f"num_bins={self.num_bins!r}: must be >= 2")
    if self.sigma_to_bin_ratio <= 0:
      problems.append(f"sigma_to_bin_ratio={self.sigma_to_bin_ratio!r}: must be > 0")
    return problems


@register_head
@dataclass(frozen=True)
class C51HeadConfig(CategoricalHeadConfig):
  class_name: str = "C51"

  def build(self, env_stats, reward_scale) -> "C51Head":
    return C51Head(self, env_stats, reward_scale)


@register_head
@dataclass(frozen=True)
class HLGaussHeadConfig(CategoricalHeadConfig):
  class_name: str = "HLGauss"

  def build(self, env_stats, reward_scale) -> "HLGaussHead":
    return HLGaussHead(self, env_stats, reward_scale)


# ---- parts -------------------------------------------------------------------


class Head(Part):
  """The representation interface `QCritic` trains through."""

  def expectation_members(self, raw):
    """(N, B, 1) float32 expected value per member."""
    raise NotImplementedError

  def bootstrap_members(self, next_raw):
    """(N, B, K) per-member quantity `td_target` consumes."""
    raise NotImplementedError

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    raise NotImplementedError

  def regression_loss(self, raw, target):
    raise NotImplementedError

  def soft_target(self, raw):
    """Ensemble-mean prediction, in the representation `regression_loss` fits."""
    raise NotImplementedError

  def target_expectation(self, target):
    raise NotImplementedError

  def record(self, metrics, raw) -> None:
    """Head-specific diagnostics, recorded on the critic's own namespace."""

  def losses(self, chunk_return, discount, bootstrap, entropy_bonus) -> dict:
    """Losses for parameters the head trains besides the network (a learned
    support), under their own optimizer keys."""
    return {}


class ScalarHead(Head):

  def expectation_members(self, raw):
    return raw.astype(jnp.float32)

  def bootstrap_members(self, next_raw):
    return self.expectation_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    return scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus)

  def regression_loss(self, raw, target):
    return ((raw.astype(jnp.float32) - target.astype(jnp.float32)) ** 2).mean()

  def soft_target(self, raw):
    return raw.mean(axis=0)

  def target_expectation(self, target):
    return target.astype(jnp.float32)


class CategoricalHead(Head):
  """Softmax over `num_bins` bins placed on the support, fit with cross-entropy.
  A subclass supplies the bin layout (`intervals`, `layout`), the target
  encoding and the leak its learned support is constrained by."""

  reports_sigma = True

  def __init__(self, config: CategoricalHeadConfig, env_stats, reward_scale):
    super().__init__(config)
    self.support = config.support.build(self, env_stats, reward_scale)

  # ---- geometry ----------------------------------------------------------------

  def intervals(self) -> int:
    raise NotImplementedError

  def layout(self, support, lo, hi):
    """`(edges or None, centers, sigma)` of the bins over `[lo, hi]`."""
    raise NotImplementedError

  def sigma(self, support, lo, hi):
    """The Gaussian width, `sigma_to_bin_ratio` bin widths."""
    return support.floor_sigma(self.config.sigma_to_bin_ratio * (hi - lo) / self.intervals())

  def geometry(self):
    """The current geometry: a constant to every loss."""
    return self.support.geometry(self)

  def support_values(self):
    """The value each bin / atom stands for, (num_bins,)."""
    return self.geometry()[1]

  def leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """Expected mass of the TD target outside `[lo, hi]` under Gaussians of width
    `sigma`, averaged over the batch; differentiable in `lo`, `hi` and `sigma`."""
    raise NotImplementedError

  # ---- representation ----------------------------------------------------------

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_members(self, raw):
    return (self.probs(raw) * self.support_values()).sum(axis=-1, keepdims=True)

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    # Summed over atoms and members, averaged over the batch.
    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()
    loss = reduce(-(target.astype(jnp.float32) * logp))
    if self.config.entropy_coef:
      loss += self.config.entropy_coef * reduce(jnp.exp(logp) * logp)
    return self.support.scale_loss(loss)

  def soft_target(self, raw):
    return self.probs(raw).mean(axis=0)

  def target_expectation(self, target):
    return (target.astype(jnp.float32) * self.support_values()).sum(axis=-1, keepdims=True)

  def record(self, metrics, raw) -> None:
    probs = self.probs(raw)
    support = self.support_values()
    exp = (probs * support).sum(axis=-1, keepdims=True)
    var = (probs * (support - exp) ** 2).sum(axis=-1)
    metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))

  def losses(self, chunk_return, discount, bootstrap, entropy_bonus) -> dict:
    return self.support.losses(self, chunk_return, discount, bootstrap, entropy_bonus)

  def derived(self) -> dict:
    learned = self.support.learned
    lo, hi = self.support.trained_bounds() if learned else self.support.bounds()
    out = {"v_min": float(lo), "v_max": float(hi),
           "bin_width": float(hi - lo) / self.intervals()}
    sigma = self.layout(self.support, lo, hi)[2] if learned else self.support.sigma
    if sigma is not None:
      out["sigma"] = float(sigma)
    out["support"] = "learned" if learned else "fixed"
    return out


class C51Head(CategoricalHead):
  """Atoms on a grid over the support; the bootstrap distribution is projected."""

  # The projection never smooths, so a fixed support has no use for sigma.
  reports_sigma = False

  def intervals(self) -> int:
    return self.config.num_bins - 1

  def layout(self, support, lo, hi):
    return None, support.grid(lo, hi, self.config.num_bins), self.sigma(support, lo, hi)

  def bootstrap_members(self, next_raw):
    return self.probs(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    lo, hi = self.support.bounds()
    return categorical_projection(bootstrap, self.support_values(), chunk_return, discount,
                                  lo, hi, entropy_bonus)

  def leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """The shifted bootstrap distribution is a mixture of point masses at
    `chunk_return + discount * (atom + bonus)`; each is read as a Gaussian of
    width `sigma`. Atoms the projection would clip count as leaked."""
    tz = chunk_return + discount * (centers[None, :] + entropy_bonus)
    outside = gaussian_mass_outside(lo, hi, tz, sigma)
    return (bootstrap.astype(jnp.float32) * outside).sum(axis=-1).mean()


class HLGaussHead(CategoricalHead):
  """Bin centers as support; the scalar TD target is smoothed into a histogram."""

  def intervals(self) -> int:
    return self.config.num_bins

  def layout(self, support, lo, hi):
    edges = support.grid(lo, hi, self.config.num_bins + 1)
    return edges, (edges[:-1] + edges[1:]) / 2, self.sigma(support, lo, hi)

  def bootstrap_members(self, next_raw):
    return self.expectation_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    edges, _, sigma = self.geometry()
    return to_histogram(scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus),
                        edges, sigma)

  def leak(self, lo, hi, sigma, centers, chunk_return, discount, bootstrap, entropy_bonus):
    """Mass of N(y, sigma^2) outside `[lo, hi]` for the unclipped scalar target y."""
    y = scalar_td_target(chunk_return, discount, bootstrap, entropy_bonus).astype(jnp.float32)
    return gaussian_mass_outside(lo, hi, y, sigma).mean()
