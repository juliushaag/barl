"""Value Forcing critic (Zheng et al., arXiv:2611.02160).

The return distribution is a flow: a velocity field `v(z | s, a, tau)` transports
`z0 ~ N(0, 1)` at noise level tau = 0 to a return sample at tau = 1. Value
Forcing trains it with a Bellman backup at *every* noise level, the flow Bellman
equation (their Eq. 5)

    Z(s, a, tau)  =d  tau * r + gamma * Z(s', a', tau) + (1 - tau) * sqrt(1 - gamma^2) * N

whose Gaussian correction keeps both sides standard normal at tau = 0 and which
reduces to the distributional Bellman equation at tau = 1. Differentiating in tau
gives the regression target for the velocity (their Eq. 13).

This is the paper's Q-velocity-only variant ("VF without the V-velocity",
App. D.5): the Q-flow bootstraps from its own target copy at the policy's next
action, which is the form an online actor-critic needs. The paper's V-velocity,
flow behavior-cloning policy and rejection sampling are offline-RL machinery and
are not part of a critic; the actor supplies `a'` here.

Generalised from the paper's one-step, non-terminating backup to this repo's
transitions: with `g = (1 - terminated) * gamma^horizon` and `R` the chunk
return,

    z_in   = tau * (R + g * bonus) + g * z_tau + (1 - tau) * sqrt(1 - g^2) * eps
    target =        R + g * bonus  + g * v_target(z_tau | s', a', tau) - sqrt(1 - g^2) * eps
    loss   = (v(z_in | s, a, tau) - target)^2

where `z_tau` is the target flow integrated from `z0` to `tau` at `(s', a')`.
A terminal transition (`g = 0`) flows straight from noise to `R`.

Returns are modelled in normalised units, `return_scale * reward_scale * G`
(`return_scale` defaults to the paper's `1 - gamma`); `value()` converts back,
so actors and entropy terms see ordinary Q-values.

Deviations from the paper, for this codebase's contracts:

* `value()` takes no random key, so the Q estimate integrates from fixed
  Gaussian quantile points instead of random draws of `z0`.
* With `target_reduce: mean` each member bootstraps from its own target copy;
  with `min` every member bootstraps from the target member whose value
  estimate at `(s', a')` is lower (clipped double Q on return flows).
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.metrics import armed
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.time_embedding import SinusoidalTimeConfig, TimeEmbeddingConfig
from barl_jax.registration import register_component

from .base import BATCH_STAT_NORMS, EnsembleConfig, QCritic, QCriticConfig
from .heads import HeadConfig, ScalarHeadConfig
from .targets import PolyakConfig, TargetConfig


@dataclass(frozen=True)
class ValueForcingCriticConfig(QCriticConfig):
  class_name: str = "ValueForcingCritic"
  latent_dim: int = 512
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(
    activation="gelu", norm="layer", norm_stem=True, norm_after_act=True))
  head: HeadConfig = field(default_factory=ScalarHeadConfig)
  ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
  target: TargetConfig = field(default_factory=lambda: PolyakConfig(joint_forward=False))
  # Euler steps used to bring the bootstrap sample from z0 to z_tau.
  flow_steps: int = 10
  # The Q estimate: `value_samples` return samples, each integrated in
  # `value_steps` Euler steps. The paper trains with the one-step estimate
  # (z0 + v(z0 | s, a, 0)) and evaluates with 10 steps; one step keeps actor
  # losses, which differentiate through this, cheap.
  value_steps: int = 1
  value_samples: int = 8
  # Returns are modelled as `return_scale * reward_scale * G`; None is 1 - gamma.
  return_scale: float | None = None
  time: TimeEmbeddingConfig = field(default_factory=lambda: SinusoidalTimeConfig(time_dim=16))

  def problems(self) -> list[str]:
    problems = super().problems()
    for name in ("flow_steps", "value_steps", "value_samples"):
      if getattr(self, name) < 1:
        problems.append(f"{name}={getattr(self, name)!r}: must be >= 1")
    if not (math.isfinite(self.gamma) and 0 <= self.gamma < 1):
      problems.append(f"gamma={self.gamma!r}: must satisfy 0 <= gamma < 1")
    if self.return_scale is not None and not (
        math.isfinite(self.return_scale) and self.return_scale > 0):
      problems.append(f"return_scale={self.return_scale!r}: must be None or finite and > 0")
    return problems


class ReturnVelocity(nnx.Module):
  """One velocity member: `(s, a, z, tau) -> dz/dtau`, a trunk over
  concat(s, a, z) conditioned on the embedded noise level."""

  def __init__(self, obs_dim, action_dim, *, latent_dim, layers,
               time: TimeEmbeddingConfig, mlp: NetworkConfig, rngs):
    self.time = time.build(rngs=rngs)
    self.net = mlp.build(obs_dim + action_dim + 1, 1, latent_dim, layers,
                         cond_dim=time.time_dim, rngs=rngs)

  def __call__(self, obs, action, z, tau):
    return self.net((obs, action, z), (self.time(tau),)).astype(jnp.float32)


class ReturnVelocityEnsemble(nnx.Module):
  """Members evaluated on a shared `(B, 1)` noisy return, or each on its own
  `(N, B, 1)` one -- an independent flow path per member."""

  def __init__(self, n_models, *args, rngs, **kwargs):
    self.members = Ensemble(n_models, ReturnVelocity, *args, rngs=rngs, **kwargs)

  def __call__(self, obs, action, z, tau):
    if z.ndim == 2:
      return self.members(obs, action, z, tau)

    @nnx.vmap(in_axes=(0, None, None, 0, None), out_axes=0)
    def call(member, obs, action, z, tau):
      return member(obs, action, z, tau)

    return call(self.members.members, obs, action, z, tau)


def gaussian_quantiles(n: int):
  """`n` standard-normal quantiles at the midpoints of `n` equal-mass bins."""
  return jax.scipy.special.ndtri((jnp.arange(n, dtype=jnp.float32) + 0.5) / n)


@register_component
class ValueForcingCritic(QCritic):
  """Flow-matching return critic trained with the flow Bellman backup. Shares
  the target part, ensemble reductions and n-step inputs with every other
  critic; the network, the Q estimate and the regression are its own, so it
  takes the scalar head and the plain TD objective only."""

  config_type = ValueForcingCriticConfig

  def __init__(self, env_stats, config, *, rngs: nnx.Rngs, **kwargs):
    self.rngs = rngs
    super().__init__(env_stats, config, rngs=rngs, **kwargs)

  @classmethod
  def config_problems(cls, config) -> list[str]:
    problems = super().config_problems(config)
    if config.head.class_name != "Scalar":
      problems.append(f"head={config.head.class_name!r}: ValueForcingCritic needs the Scalar head")
    if config.objective.class_name != "TD":
      problems.append(f"objective={config.objective.class_name!r}: "
                      "ValueForcingCritic trains on TD only")
    if config.mlp.norm in BATCH_STAT_NORMS:
      problems.append(f"mlp.norm={config.mlp.norm!r}: the flow backup evaluates (s, a) and "
                      "(s', a') at different noisy returns, so they cannot share batch statistics")
    return problems

  def build_network(self, env_stats, config, *, rngs):
    return ReturnVelocityEnsemble(
      config.ensemble.n_models, env_stats["obs_dim"], env_stats["act_dim"],
      latent_dim=config.latent_dim, layers=config.layers, time=config.time, mlp=config.mlp,
      rngs=rngs)

  @property
  def return_scale(self) -> float:
    scale = self.config.return_scale
    return float(1.0 - self.config.gamma if scale is None else scale)

  def derived(self) -> dict:
    return {**super().derived(), "return_scale": self.return_scale}

  # ---- the flow ------------------------------------------------------------

  def integrate(self, net, obs, action, z0, tau_end, steps: int):
    """Euler-integrate `z0` (B, 1) from noise level 0 to `tau_end` (B, 1) or a
    scalar; one path per member, `(N, B, 1)`."""
    tau_end = jnp.broadcast_to(jnp.asarray(tau_end, jnp.float32), z0.shape)
    z = jnp.broadcast_to(z0, (self.n_models, *z0.shape))
    step = tau_end / steps
    for i in range(steps):
      z = z + step * net(obs, action, z, i * step)
    return z

  def _estimate(self, net, obs, action, samples: int, steps: int):
    """Mean return per member in normalised units, `(N, B, 1)`: `samples`
    Gaussian-quantile starts integrated to tau = 1."""
    b = obs.shape[0]
    z0 = jnp.tile(gaussian_quantiles(samples)[:, None], (b, 1))        # (B * M, 1)
    z1 = self.integrate(net, jnp.repeat(obs, samples, axis=0), jnp.repeat(action, samples, axis=0),
                        z0, 1.0, steps)
    return z1.reshape(self.n_models, b, samples, 1).mean(axis=2)

  def raw(self, s, a, level=None):
    """Per-member Q in reward-scaled units, `(N, B, 1)`."""
    estimate = self._estimate(self.net, s, a, self.config.value_samples, self.config.value_steps)
    return estimate / self.return_scale

  def return_samples(self, s, a, key, samples: int = 32, steps: int | None = None):
    """Samples of the return distribution in reward-scaled units, `(N, B, samples)`."""
    b = s.shape[0]
    z0 = jax.random.normal(key, (b * samples, 1))
    z1 = self.integrate(self.net, jnp.repeat(s, samples, axis=0), jnp.repeat(a, samples, axis=0),
                        z0, 1.0, self.config.flow_steps if steps is None else steps)
    return z1.reshape(self.n_models, b, samples) / self.return_scale

  # ---- the flow Bellman backup ---------------------------------------------

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    key = self.rngs.noise() if key is None else key
    tau_key, start_key, eps_key = jax.random.split(key, 3)
    b = obs.shape[0]
    tau = jax.random.uniform(tau_key, (b, 1))
    z0 = jax.random.normal(start_key, (b, 1))
    eps = jax.random.normal(eps_key, (b, 1))

    # The bootstrap flow at (s', a'): its noisy return at tau and its velocity there.
    target_net = self.target.bootstrap_net(self.net)
    z_tau = self.integrate(target_net, next_obs, next_action, z0, tau, self.config.flow_steps)
    v_next = target_net(next_obs, next_action, z_tau, tau)
    next_estimate = self._estimate(target_net, next_obs, next_action,
                                   self.config.value_samples, self.config.value_steps)
    if self.config.ensemble.target_reduce == "min":
      idx = jnp.argmin(next_estimate, axis=0, keepdims=True)
      pick = lambda x: jnp.broadcast_to(jnp.take_along_axis(x, idx, axis=0), x.shape)
      z_tau, v_next = pick(z_tau), pick(v_next)
    z_tau, v_next, next_estimate = jax.lax.stop_gradient((z_tau, v_next, next_estimate))

    scale = self.return_scale
    chunk_return, discount = self.n_step_inputs(chunk_return, terminated, horizon)
    drift = scale * (chunk_return + discount * jax.lax.stop_gradient(entropy_bonus))
    spread = jnp.sqrt(jnp.maximum(1.0 - discount ** 2, 0.0))

    z_in = tau * drift + discount * z_tau + (1.0 - tau) * spread * eps
    target = jax.lax.stop_gradient(drift + discount * v_next - spread * eps)
    pred = self.net(obs, action, z_in, tau)
    # Mean over the batch, summed over members.
    loss = ((pred - target) ** 2).mean(axis=(1, 2)).sum()

    if armed():
      bootstrap = self.bootstrap(next_estimate) / scale
      self.record(jax.lax.stop_gradient(self.raw(obs, action)),
                  chunk_return + discount * (bootstrap + entropy_bonus))
      self.metrics.dist("velocity", pred.mean(axis=0))
    return {"critic": loss}
