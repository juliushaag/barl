"""Bootstrap targets: which network prices `(s', a')`, and how the live and
bootstrap forwards are batched.

- `NoTarget`  bootstraps through the frozen live network (CrossQ). With
  `joint_forward` the `(s, a)` and `(s', a')` halves share one forward, so
  batch-statistic norms see both.
- `Polyak`    keeps a slowly blended copy of the network (`tau`, every `delay`
  steps). With `joint_forward` and a batch-statistic norm, the target copy also
  sees the joint batch (XQC).
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer, Part, frozen
from barl_jax.config.serialization import variant_registry

TARGETS, register_target = variant_registry()


@dataclass(frozen=True)
class TargetConfig:
  variants: ClassVar[dict[str, type]] = TARGETS

  class_name: str = ""
  joint_forward: bool = True

  def problems(self) -> list[str]:
    return []


@register_target
@dataclass(frozen=True)
class NoTargetConfig(TargetConfig):
  class_name: str = "NoTarget"

  def build(self, net) -> "NoTarget":
    return NoTarget(self)


@register_target
@dataclass(frozen=True)
class PolyakConfig(TargetConfig):
  class_name: str = "Polyak"
  tau: float = 0.005
  delay: int = 1

  def problems(self) -> list[str]:
    problems = []
    if not 0 < self.tau <= 1:
      problems.append(f"tau={self.tau!r}: must be in (0, 1]")
    if self.delay < 1:
      problems.append(f"delay={self.delay!r}: must be >= 1")
    return problems

  def build(self, net) -> "Polyak":
    return Polyak(self, net)


def joint_forward(net, items):
  """One forward over the concatenated `(s, a, level)` items, split back per item."""
  cat = lambda i: None if items[0][i] is None else jnp.concatenate([x[i] for x in items])
  joint = net(cat(0), cat(1), cat(2))
  bounds = [0]
  for s, _, _ in items:
    bounds.append(bounds[-1] + s.shape[0])
  return [joint[..., lo:hi, :] for lo, hi in zip(bounds[:-1], bounds[1:])]


class Target(Part):
  """Plans the live and bootstrap forwards for one TD step."""

  def forward(self, net, items, next_item, *, batch_stats: bool):
    """`(live predictions per item, stop-gradient bootstrap output)`.
    `batch_stats`: whether the network normalizes with batch statistics."""
    raise NotImplementedError

  def bootstrap_net(self, net):
    """The network that prices `(s', a')` for a caller doing its own forwards."""
    raise NotImplementedError

  def sync(self, net) -> None:
    """Track the live network after an optimizer step."""


class NoTarget(Target):

  def forward(self, net, items, next_item, *, batch_stats: bool):
    if self.config.joint_forward:
      *preds, next_raw = joint_forward(net, items + [next_item])
    else:
      preds = joint_forward(net, items)
      next_raw = frozen(net)(*next_item)
    return preds, jax.lax.stop_gradient(next_raw)

  def bootstrap_net(self, net):
    return net


class Polyak(Target):

  def __init__(self, config: PolyakConfig, net):
    super().__init__(config)
    self.net_target = nnx.clone(net)
    self.steps = Buffer(jnp.zeros((), jnp.int32))

  def forward(self, net, items, next_item, *, batch_stats: bool):
    # Sharing one batch only matters for batch-statistic norms; otherwise the
    # extra passes over the other halves are pure waste.
    if self.config.joint_forward and batch_stats:
      *preds, _ = joint_forward(net, items + [next_item])
      next_raw = joint_forward(self.net_target, items + [next_item])[-1]
    else:
      preds = joint_forward(net, items)
      next_raw = self.net_target(*next_item)
    return preds, jax.lax.stop_gradient(next_raw)

  def bootstrap_net(self, net):
    return self.net_target

  def sync(self, net) -> None:
    """Blend towards `net` by `tau`, every `delay`-th call."""
    self.steps[...] += 1
    live = nnx.state(net, nnx.Param)
    tgt = nnx.state(self.net_target, nnx.Param)
    blend = lambda t, l: t + self.config.tau * (l - t)
    if self.config.delay > 1:
      due = (self.steps[...] % self.config.delay) == 0
      step = jax.tree.map(lambda t, l: jnp.where(due, blend(t, l), t), tgt, live)
    else:
      step = jax.tree.map(blend, tgt, live)
    nnx.update(self.net_target, step)
    self.metrics.gap("", live, tgt)

  def reset_counters(self):
    self.steps[...] = jnp.zeros_like(self.steps[...])
