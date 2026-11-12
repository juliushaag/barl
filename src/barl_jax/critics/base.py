"""The composable Q critic.

`QCritic` owns the network and the TD loop; everything else is a part, chosen by
its own `class:` and documented in its module:

  head       Scalar | C51 | HLGauss       what the outputs represent (`heads.py`)
  head.support  FixedSupport | LearnedSupport   categorical heads only (`supports.py`)
  target     NoTarget | Polyak            who prices (s', a') (`targets.py`)
  objective  TD | NoiseConditionedTD | CQL   what the network is trained on (`objectives.py`)
  ensemble   n_models, value/target reduce (mean | min)
  mlp        the network trunk (`barl_jax.networks`)
  constraints  e.g. [WeightNorm] projections after each step (`networks/constraints.py`)

A part's config path, state path and metric namespace coincide: `head.support`
builds `critic.head.support` and logs under `critic/head/support/*`.

Combinations a part cannot serve are stated once, in `COMBINATION_RULES` (and,
for a critic that replaces machinery, in its own `config_problems`), and are
reported before anything is built.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Critic
from barl_jax.critics.mlps import crossq_mlp
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.constraints import apply_constraints
from barl_jax.networks.ensemble import Ensemble
from barl_jax.optimization import CriticOptimConfig
from barl_jax.registration import register_component

from .heads import HeadConfig, ScalarHeadConfig
from .objectives import ObjectiveConfig, TDConfig
from .targets import NoTargetConfig, Polyak, TargetConfig

REDUCTIONS = ("mean", "min")
BATCH_STAT_NORMS = ("batch", "batchrenorm")


@dataclass(frozen=True)
class EnsembleConfig:
  n_models: int = 2
  value_reduce: str = "mean"
  target_reduce: str = "mean"

  def problems(self) -> list[str]:
    problems = [f"{name}={getattr(self, name)!r} not in {REDUCTIONS}"
                for name in ("value_reduce", "target_reduce")
                if getattr(self, name) not in REDUCTIONS]
    if self.n_models < 1:
      problems.append(f"n_models={self.n_models!r}: must be >= 1")
    return problems


@dataclass(frozen=True)
class QCriticConfig(CriticOptimConfig):
  class_name: str = "QCritic"
  gamma: float = 0.99
  reward_scale: float = 1.0
  latent_dim: int = 2048
  layers: int = 2
  mlp: NetworkConfig = field(default_factory=crossq_mlp)
  head: HeadConfig = field(default_factory=ScalarHeadConfig)
  target: TargetConfig = field(default_factory=NoTargetConfig)
  ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
  objective: ObjectiveConfig = field(default_factory=TDConfig)


def _batch_stats_need_joint_forward(config):
  if config.mlp.norm in BATCH_STAT_NORMS and not config.target.joint_forward:
    return (f"mlp.norm={config.mlp.norm!r} needs target.joint_forward: batch statistics "
            "must be shared between the (s,a) and (s',a') halves")
  return None


def _cql_needs_stat_free_network(config):
  if config.objective.class_name == "CQL" and config.mlp.norm in BATCH_STAT_NORMS:
    return (f"objective=CQL with mlp.norm={config.mlp.norm!r}: the extra action forward "
            "would pollute the batch statistics")
  return None


# Cross-part rules: each returns a problem string or None.
COMBINATION_RULES = (_batch_stats_need_joint_forward, _cql_needs_stat_free_network)


def n_step_discount(gamma, horizon, terminated, dtype):
  """(1 - terminated) * gamma**horizon, shaped (B, 1). `horizon` is the
  executed plan length and is always passed explicitly."""
  horizon = jnp.asarray(horizon, dtype).reshape(-1, 1)
  terminated = terminated.reshape(-1, 1).astype(dtype)
  return (1.0 - terminated) * jnp.power(jnp.asarray(gamma, dtype), horizon)


def reduce_members(x, expectation, kind: str):
  """Reduce the member axis of `x` (N, B, K). `min` selects, per sample, the
  whole member with the lowest `expectation` (N, B, 1) -- a scalar min for a
  scalar `x`, the lower head's distribution for a categorical one."""
  if kind == "mean":
    return x.mean(axis=0)
  idx = jnp.argmin(expectation, axis=0, keepdims=True)
  return jnp.take_along_axis(x, idx, axis=0)[0]


class QNetwork(nnx.Module):
  """An `Ensemble` of trunks over concat(s, a[, encoder(level)]). The level
  encoder lives inside so a target clone carries its own copy."""

  def __init__(self, obs_dim: int, act_dim: int, out_dim: int, config: QCriticConfig,
               *, rngs: nnx.Rngs):
    time = config.objective.conditioning()
    self.encoder = time.build(rngs=rngs) if time else None
    self.ensemble = Ensemble(config.ensemble.n_models, config.mlp.build, obs_dim + act_dim,
                             out_dim, config.latent_dim, config.layers,
                             cond_dim=time.time_dim if time else 0, rngs=rngs)

  def __call__(self, s, a, level=None):
    cond = ()
    if self.encoder is not None:
      cond = (self.encoder(level.astype(jnp.float32)[:, None]),)
    return self.ensemble((s, a), cond)


@register_component
class QCritic(Critic):
  """Network + parts + the shared TD step. Raw outputs are always
  `(n_models, B, out_dim)`; the head interprets them."""

  config_type = QCriticConfig

  def __init__(self, env_stats, config, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))

    self.head = config.head.build(env_stats, config.reward_scale, gamma=config.gamma)
    self.objective = config.objective.build(env_stats)
    self.net = self.build_network(env_stats, config, rngs=rngs)
    # The owning Policy applies constraints after construction too; doing it
    # here as well means a target network starts from the projected weights.
    apply_constraints(self)
    self.target = config.target.build(self.net)

  @classmethod
  def config_problems(cls, config) -> list[str]:
    problems = super().config_problems(config)
    problems += [p for rule in COMBINATION_RULES if (p := rule(config))]
    return problems

  @classmethod
  def learning_objective_for(cls, config) -> str:
    return config.objective.learning_objective

  def build_network(self, env_stats, config, *, rngs):
    return QNetwork(env_stats["obs_dim"], env_stats["act_dim"], config.head.out_dim(), config,
                    rngs=rngs)

  @property
  def has_target(self) -> bool:
    return isinstance(self.target, Polyak)

  @property
  def n_models(self) -> int:
    return self.config.ensemble.n_models

  @property
  def supports_bon(self) -> bool:
    return self.objective.supports_bon

  @property
  def policy_samples(self) -> int:
    """Policy actions per state `td_loss` needs as `policy_samples`."""
    return self.objective.policy_samples

  def create_optimizer(self) -> dict:
    return {"critic": self.config.build(), **super().create_optimizer()}

  # ---- head interface, delegated ---------------------------------------------

  def expectation_members(self, raw):
    return self.head.expectation_members(raw)

  def bootstrap_members(self, next_raw):
    return self.head.bootstrap_members(next_raw)

  def td_target(self, chunk_return, discount, bootstrap, entropy_bonus):
    return self.head.td_target(chunk_return, discount, bootstrap, entropy_bonus)

  def regression_loss(self, raw, target):
    return self.head.regression_loss(raw, target)

  def soft_target(self, raw):
    return self.head.soft_target(raw)

  def target_expectation(self, target):
    return self.head.target_expectation(target)

  def support_values(self):
    """The value each bin / atom of a categorical head stands for."""
    return self.head.support_values()

  # ---- pricing -----------------------------------------------------------

  def _level(self, s, level):
    return self.objective.level(s, level)

  def raw(self, s, a, level=None):
    return self.net(s, a, self._level(s, level))

  def value(self, s, a, level=None):
    return self.reduce_value(self.raw(s, a, level))

  def expectation(self, s, a, level=None):
    return self.expectation_members(self.raw(s, a, level)).mean(axis=0)

  def reduce_value(self, raw):
    exp = self.expectation_members(raw)
    return reduce_members(exp, exp, self.config.ensemble.value_reduce)

  def pessimism_gap(self, raw):
    return self.expectation_members(raw).mean(axis=0) - self.reduce_value(raw)

  # ---- TD ----------------------------------------------------------------

  def bootstrap(self, next_raw):
    return reduce_members(self.bootstrap_members(next_raw), self.expectation_members(next_raw),
                          self.config.ensemble.target_reduce)

  def n_step_inputs(self, chunk_return, terminated, horizon):
    """`(reward-scaled chunk return, discount)`, both (B, 1)."""
    chunk_return = chunk_return.reshape(-1, 1) * self.config.reward_scale
    return chunk_return, n_step_discount(self.config.gamma, horizon, terminated, chunk_return.dtype)

  def bellman_target(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0):
    chunk_return, discount = self.n_step_inputs(chunk_return, terminated, horizon)
    return jax.lax.stop_gradient(self.td_target(chunk_return, discount, bootstrap, entropy_bonus))

  def head_losses(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0) -> dict:
    """Losses for parameters the head trains besides the network (a learned
    support), keyed by their own optimizer keys."""
    chunk_return, discount = self.n_step_inputs(chunk_return, terminated, horizon)
    return self.head.losses(chunk_return, discount, bootstrap, entropy_bonus)

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None, policy_samples=None) -> dict:
    """chunk_return: (B,) return already discounted within the executed
    horizon; terminated: (B,) true termination (never truncation); horizon:
    (B,) executed plan length; entropy_bonus: (B, 1) or 0.0, added to the
    bootstrap before discounting. policy_samples: `(curr, curr_logp, next,
    next_logp)` when the objective asks for them (`self.policy_samples`)."""
    items = [(obs, action, self._level(obs, None))]
    items += self.objective.extra_items(obs, action, key)
    next_item = (next_obs, next_action, self._level(next_obs, None))

    preds, next_raw = self.target.forward(self.net, items, next_item,
                                          batch_stats=self.config.mlp.norm in BATCH_STAT_NORMS)
    bootstrap = self.bootstrap(next_raw)
    target = self.bellman_target(chunk_return, terminated, horizon, bootstrap, entropy_bonus)
    loss = self.regression_loss(preds[0], target)
    self.record(preds[0], target)
    loss = self.objective.loss(self, loss, preds, obs=obs, next_obs=next_obs, key=key,
                               policy_samples=policy_samples)
    return {"critic": loss,
            **self.head_losses(chunk_return, terminated, horizon, bootstrap, entropy_bonus)}

  def record(self, raw, target) -> None:
    exp = self.expectation_members(raw)
    self.metrics.dist("q", exp.mean(axis=0))
    self.metrics.dist("q_target", self.target_expectation(target))
    self.metrics.ensemble("ensemble", exp, gap=self.pessimism_gap(raw))
    self.head.record(self.metrics, raw)

  def after_step(self):
    super().after_step()
    # After the Policy applied `constraints`: the target tracks the projected weights.
    self.target.sync(self.net)
