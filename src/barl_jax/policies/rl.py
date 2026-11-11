"""The shared off-policy actor-critic template.

SAC, AMDP and DIME's `train_step`s were, at bottom, one function: a
stop-gradiented bootstrap draw at `next_obs`, a critic Bellman target with an
optional entropy bonus, an actor loss gated at some cadence, and an entropy
dual fed the bootstrap's entropy. What differed was four things -- obs
normalisation, the gate's cadence, the name of the actor's generator, and
whether the dual is fed anything at all -- and none of those is an algorithm.

`RLPolicy` states that shared step once; `MaxEntRLPolicy` adds the dual. Every
algorithm reuses this template: `SAC`, `AMDP` and `DIME` (each beside its
actor, in `barl_jax.actors.*`) are thin `MaxEntRLPolicy` subclasses that pin
`actor_cls`, the actor-policy integration binding for that algorithm; `NCLQL`
and `SDAC` pin one on `RLPolicy` directly (no entropy dual). The algorithm
difference is entirely in which actor/critic/params a model variant's
`ModelConfig` selects and, for SDAC, the `actor_objective` seam, never in a
second copy of this training step."""

import math
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp

from barl_jax.components import ActionSampler
from barl_jax.config.schemas import LR_DECAYS
from barl_jax.critics import QCritic
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.optimization import gated_losses
from barl_jax.policy import Policy
from barl_jax.registration import register_policy


@dataclass(frozen=True)
class PolicyConfig:
  """Params every off-policy actor-critic policy owns."""

  normalize_obs: bool = False
  policy_delay: int = 1
  random_steps: int = 5000
  batch_size: int = 256
  utd_ratio: float = 2.0
  lr_warmup_steps: int = 0
  lr_decay: str = "constant"


@dataclass(frozen=True)
class MaxEntPolicyConfig(PolicyConfig):
  entropy_scale: float = -1.0
  entropy_in_target: bool = True


@register_policy
class RLPolicy(Policy):
  """Actor + critic, no entropy dual -- `alpha`/`target_entropy_bonus`/`dual_loss`
  are the three seams `MaxEntRLPolicy` fills in. On its own this is TD3-shaped:
  nothing in this repo's config points a model at `RLPolicy` directly, but it
  is the layout every actor-critic algorithm here shares, and an actor that
  adds no entropy term to its own objective needs no dual at all."""

  SLOTS = ("critic", "actor")
  @classmethod
  def slot_objective(cls, slot, actor_cls=None):
    if slot == "critic" and actor_cls is not None:
      return cls.SLOT_OBJECTIVES.get(slot, actor_cls.critic_objective)
    return super().slot_objective(slot, actor_cls)

  # The actor-policy integration binding: a per-algorithm subclass (`SAC`,
  # `AMDP`, `DIME`, `SDAC`, `NCLQL`, each beside its actor implementation) sets
  # this to its concrete actor class, tightening build_component's gate from
  # the generic ActionSampler ABC. A `components.actor.class` mismatch then
  # fails at construction with a clear error naming both classes, instead of
  # silently accepting whatever the config named.
  actor_cls: type = ActionSampler

  def __init__(self, env: gym.Env, *args, **kwargs):
    super().__init__(env, *args, **kwargs)
    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic: QCritic = self.build_component("critic", QCritic)
    self.actor: ActionSampler = self.build_component("actor", self.actor_cls)
    # No cadence/optimizer-key guard here: `self.optimizer_cadence()` is a
    # virtual call, and a MaxEntRLPolicy subclass hasn't built `self.entropy`
    # yet at this point in `super().__init__()` -- the check belongs at the
    # end of whichever `__init__` runs last (MaxEntRLPolicy's, today).

  @property
  def policy_delay(self) -> int:
    """Grad steps between actor/dual updates (the critic steps every step)."""
    return self.params.policy_delay

  @classmethod
  def config_problems(cls, params) -> list[str]:
    problems = []
    if params.policy_delay < 1:
      problems.append(f"policy_delay={params.policy_delay!r}: must be >= 1")
    if params.random_steps < 0:
      problems.append(f"random_steps={params.random_steps!r}: must be >= 0")
    if params.batch_size < 1:
      problems.append(f"batch_size={params.batch_size!r}: must be >= 1")
    if not (math.isfinite(params.utd_ratio) and params.utd_ratio > 0):
      problems.append(f"utd_ratio={params.utd_ratio!r}: must be a finite number > 0")
    if params.lr_warmup_steps < 0:
      problems.append(f"lr_warmup_steps={params.lr_warmup_steps!r}: must be >= 0")
    if params.lr_decay not in LR_DECAYS:
      problems.append(f"lr_decay={params.lr_decay!r}: must be one of {LR_DECAYS}")
    return problems

  def optimizer_cadence(self) -> dict:
    return {key: self.policy_delay for key in self.actor.create_optimizer()}

  def actor_objective(self, actor, critic, obs, action, alpha, key):
    return actor.loss(obs, critic, alpha, key=key)

  # ---- the three seams MaxEntRLPolicy fills in -----------------------------

  def alpha(self):
    return jnp.asarray(1.0)

  def target_entropy_bonus(self, alpha, entropy):
    return 0.0

  def dual_loss(self, entropy) -> dict:
    return {}

  # ---------------------------------------------------------------------

  def train_step(self, batch):
    obs          = self.normalize_observations(batch["obs"])
    next_obs     = self.normalize_observations(batch["next_obs"])

    action       = batch["act"]
    chunk_return = batch["chunk_return"]
    terminated   = batch["term"]
    horizon      = batch["horizon"]

    losses = {}
    alpha  = self.alpha()

    actor_keys = tuple(sorted(self.actor.create_optimizer()))
    # Critic-only samplers retain the original bootstrap/smoothing split.
    keys = jax.random.split(self.rngs.noise(), 3 if actor_keys else 2)

    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, critic=self.critic, key=keys[0]))

    extra = {}
    if self.critic.uses_cql:
      extra["cql_samples"] = self.cql_samples(obs, next_obs, self.rngs.noise())
    losses.update(self.critic.td_loss(
      obs, action, next_obs, next_action, chunk_return, terminated, horizon,
      entropy_bonus=self.target_entropy_bonus(alpha, entropy), key=keys[-1], **extra))

    # An actor can own multiple optimizers. Compute their shared loss whenever any partition is due; mask each
    # optimizer independently according to optimizer_cadence().
    if actor_keys:
      losses.update(gated_losses(
        self,
        lambda actor, critic: self.actor_objective(actor, critic, obs, action, alpha, keys[1]),
        actor_keys, self.actor, self.critic))
    losses.update(self.dual_loss(entropy))

    self.actor.metrics.vec("action", next_action, limit=1.0)
    if entropy is not None:
      self.actor.metrics.dist("entropy", entropy)

    return losses

  def cql_samples(self, obs, next_obs, key):
    """`(curr, curr_logp, next, next_logp)`: `cql.n_actions` policy actions (n, B, A)
    and their log-densities (n, B) at `obs` and `next_obs`, for the CQL penalty."""
    n, b = self.critic.config.cql.n_actions, obs.shape[0]

    def draw(states, key):
      repeated = jnp.broadcast_to(states, (n,) + states.shape).reshape(n * b, -1)
      action, entropy = self.actor.generate_action(repeated, critic=self.critic, key=key)
      if entropy is None:
        raise ValueError("CQL needs an actor that reports its sample entropy")
      return jax.lax.stop_gradient((action.reshape(n, b, -1), -entropy.reshape(n, b)))

    curr_key, next_key = jax.random.split(key)
    return (*draw(obs, curr_key), *draw(next_obs, next_key))

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_observations(observation)
    return self.actor.sample(obs, critic=self.critic, key=key), {}

  # ---- Best-of-N scoring (barl_jax.executors) --------------------------

  @property
  def supports_bon(self):
    return self.critic.supports_bon

  def score(self, observation, action):
    """`critic.value()` at the online critic, reduced per
    `ensemble.value_reduce`."""
    obs = self.normalize_observations(observation)
    return self.critic.value(obs, action)


@register_policy
class MaxEntRLPolicy(RLPolicy):
  """`RLPolicy` plus an automatic entropy dual -- SAC, AMDP and DIME all
  resolve here. `entropy_in_target` toggles the Bellman target's entropy bonus."""

  SLOTS = ("critic", "actor", "entropy")
  SLOT_OBJECTIVES = {"critic": "td"}

  def __init__(self, env: gym.Env, *args, **kwargs):
    super().__init__(env, *args, **kwargs)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.params.entropy_scale)
    assert set(self.optimizer_cadence()) <= set(self.create_optimizer()), (
      f"{self.model_name}: optimizer_cadence names a key create_optimizer() "
      f"doesn't -- build_optimizer's cadence.get(key, 1) would drop it silently.")

  def optimizer_cadence(self) -> dict:
    return {**super().optimizer_cadence(),
            **{key: self.policy_delay for key in self.entropy.create_optimizer()}}

  def alpha(self):
    return self.entropy.value()

  def target_entropy_bonus(self, alpha, entropy):
    """`entropy` is `(B,)`; the Bellman target wants the `(B, 1)` bonus."""
    if not self.params.entropy_in_target:
      return 0.0
    return jax.lax.stop_gradient(alpha * entropy)[:, None]

  def dual_loss(self, entropy) -> dict:
    return self.entropy.loss(entropy)
