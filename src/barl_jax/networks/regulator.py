import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.components import Component
from barl_jax.optimization import DualOptimConfig
from barl_jax.registration import register_component


@dataclass(frozen=True)
class AutotuneRegulatorConfig(DualOptimConfig):
  class_name: str = "AutotuneRegulator"
  alpha_init: float = 1e-4
  tune: bool = True


@register_component
class AutotuneRegulator(Component, nnx.Module):
  """A Lagrange multiplier tuned to hold some measured quantity at `target`
  (SAC's entropy temperature, and the same machinery for any other dual). The
  loss is `log_alpha * (value - target)`, whose gradient wrt log_alpha is
  `value - target`, so under descent alpha *falls* while the measured value sits
  above target and rises while it sits below -- the sign canonical SAC needs, and
  the opposite of what this docstring claimed before.

  `target` is always the owning Policy's own computed value, never a
  config/tunable -- so it stays a real keyword argument and is not a `Config`
  declaration. It is a plain float (SAC/AMDP/DIME: `target=self.action_dim *
  self.entropy_scale`).

  The optimizer knobs come from `DualOptimConfig` (a higher LR)."""

  config_type = AutotuneRegulatorConfig

  # False pins alpha at `alpha_init` -- `components.<slot>.tune: false` in
  # config. `create_optimizer()` still declares this dual's key (train_step's
  # loss dict must match `create_optimizer()` exactly regardless), but backed
  # by `optax.set_to_zero()` instead of Adam, so `log_alpha` still gets a
  # gradient computed and logged, it is just never applied -- alpha is a
  # config-set constant, not a silently-frozen-at-whatever-init-happened-to-be
  # one.

  # A dual has no tier ladder -- `kind` exists so the resolver can tell a sized
  # component from a non-sized one.
  kind = "dual"

  # value() does not clamp: a floor on alpha injects a constant entropy bonus into
  # the Bellman target that a near-zero-reward env cannot outvote.

  def __init__(self, env_stats=None, config=None, *, target,
               rngs: nnx.Rngs = None, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    target = jnp.asarray(target, jnp.float32)
    # 0-dim, not shape (1,): a singleton dimension makes every product with
    # alpha non-scalar, and jax.grad rejects a non-scalar loss.
    self.log_alpha = nnx.Param(jnp.asarray(math.log(config.alpha_init), jnp.float32))

    # A bare Python float, kept out of the pytree entirely.
    self.target_value = float(target)

  def value(self):
    return jnp.exp(self.log_alpha[...])

  def loss(self, value) -> dict:
    value = jax.lax.stop_gradient(value)
    alpha = self.value()
    target = self.target_value
    loss = alpha * (value - target).mean()

    alpha_sg = jax.lax.stop_gradient(alpha)
    self.metrics.scalar("alpha", alpha_sg)
    self.metrics.scalar("target", target)
    self.metrics.dist("value", value)
    self.metrics.dist("value_scaled", value * alpha_sg)
    return {f"{self.slot}_alpha": loss}

  def create_optimizer(self) -> dict:
    # Keyed by slot, so a policy's two regulators stay distinct without a
    # `name` kwarg.
    tx = self.config.build() if self.config.tune else optax.set_to_zero()
    return {f"{self.slot}_alpha": tx}
