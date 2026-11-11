"""Component contracts and NNX state utilities."""
from abc import ABC, abstractmethod

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.env import EnvStats
from barl_jax.metrics import Recorder


class Buffer(nnx.Variable):
  """Non-trainable state: normalization statistics, action bounds, target-net
  weights. Part of the checkpoint and threaded through jit like any other
  state, but never handed to an optimizer (see TRAINABLE)."""


TARGET_SUFFIX = "_target"


def is_target(path, _value) -> bool:
  """An nnx filter: does this parameter belong to a frozen target copy?

  Target nets are `nnx.clone`s and their leaves stay `nnx.Param` -- nnx.update()
  preserves a Variable's original type, so they cannot be retyped into something
  a type filter would skip. The naming convention carries the information, so it
  is stated once here rather than spelled out at each use."""
  return any(str(p).endswith(TARGET_SUFFIX) for p in path)


# What every optimizer differentiates and updates.
TRAINABLE = nnx.All(nnx.Param, nnx.Not(is_target))


def sanitize_action(action, xp=jnp):
  """Force a canonical action into the contract every consumer already assumes:
  finite, and inside [-1, 1] per dimension.

  A failsafe, not a fix. A NaN action reaching `env.step()` is unrecoverable --
  MuJoCo either raises or silently poisons the physics state, and the run dies
  thousands of steps later somewhere unrelated -- so the cost of clamping is
  bounded and the cost of not clamping is not. NaN maps to 0 (the neutral
  action) rather than to a bound, since a bound is a *decision* and 0 is the
  absence of one; infinities saturate to the bound they were heading for.

  `xp` selects the array module: the rollout loop scales host-side in numpy and
  must not be pulled back onto the device to do this. Both modules implement
  `nan_to_num`/`clip` identically here."""
  return xp.clip(xp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def action_bounds_numpy(action_space) -> tuple:
  """Host-side `(scale, bias)` derived from the env's action space."""
  if isinstance(action_space, gym.spaces.Discrete):
    return None, None
  low = np.asarray(action_space.low, dtype=np.float32)
  high = np.asarray(action_space.high, dtype=np.float32)
  return (high - low) / 2, (high + low) / 2


def frozen(module):
  view = jax.lax.stop_gradient(module)
  view.eval()
  return view


def entropy_regularized_loss(actor, obs, critic, alpha, *, key=None) -> dict:
  """The max-entropy actor objective, `-(Q(s, a) + alpha * H).mean()` at a
  reparameterized sample: `alpha` and the critic are held fixed."""
  action, entropy = actor.generate_action(obs, key=key)
  q = frozen(critic).value(obs, action)
  return {"actor": -(q + jax.lax.stop_gradient(alpha) * entropy).mean()}


class _Static:
  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return f"_Static({self.value!r})"


_INTERNED: dict = {}


def interned_static(value) -> _Static:
  """One identity-hashed wrapper per distinct `value`. A GraphDef passed as a jit
  static argument is re-hashed structurally on every call (~40% of an env step's
  host time); the wrapper makes that an identity hash, and interning keeps equal
  graphdefs on one jit cache entry."""
  static = _INTERNED.get(value)
  if static is None:
    static = _INTERNED[value] = _Static(value)
  return static


class Component:
  """Mixin for anything `Policy.build_component()` constructs.

  Takes the run's `EnvStats` and its typed config as the first two positional
  arguments and keeps only what every component wants from the former
  (`obs_dim` and `action_dim`).

  The whole config is kept as `self.config` -- read `self.config.foo` from any
  method rather than copying fields onto flat attributes. It sits behind
  `_Static` (identity-hashed) because config fields are mutable containers
  (`MLPConfig.norm_kwargs`), and nnx needs a hashable static leaf for the split
  GraphDef `jax.jit` keys its cache on. `config` is settable, so a test can
  perturb one field with `c.config = replace(c.config, gamma=0.5)`.

  The owning Policy fans `before_step`/`after_step` out to every slot, and each
  component has a `metrics` sink namespaced to its slot: `self.metrics.dist("critic",
  pred)` lands at `critic/q/*` without the component naming itself."""

  config_type = None
  learning_objective = None

  @classmethod
  def learning_objective_for(cls, config) -> str | None:
    """The training objective this component serves under `config`."""
    return cls.learning_objective

  def __init__(self, env_stats: EnvStats = None, config=None, **kwargs):
    # Neither env_stats nor config is forwarded: nnx.Module's __init__ is
    # object's, taking no args. A component that needs nothing beyond what
    # Component itself reads (e.g. TanhSquash) just never reads self.config.
    super().__init__()
    env_stats = env_stats if env_stats is not None else EnvStats()
    self.obs_dim    = env_stats.get("obs_dim")
    self.action_dim = env_stats.get("act_dim")
    # Both are overwritten by build_component with the real slot; the defaults
    # only keep a component constructed outside a Policy usable.
    self.slot       = type(self).__name__.lower()
    self._config_s  = _Static(config)

  @property
  def config(self):
    return self._config_s.value

  @config.setter
  def config(self, value):
    self._config_s.value = value

  @property
  def metrics(self) -> Recorder:
    """Diagnostics sink. A facade bound to this module, rebuilt per access and
    holding no state -- the logged key comes from where this component sits in
    the module tree, so it never names itself."""
    return Recorder(self)

  @classmethod
  def config_problems(cls, config) -> list[str]:
    """Why `config` cannot build this component: its own and its nested configs'
    `problems()`."""
    from barl_jax.config.serialization import config_problems
    return config_problems(config)

  def before_step(self):
    """Called once per gradient step, before the Policy's train_step."""
    ...

  def after_step(self):
    """Called once per gradient step, after every optimizer has stepped."""
    ...

  def create_optimizer(self) -> dict:
    """Optimizer key -> optax transform. The keys must match the keys this
    component's owning Policy returns from train_step."""
    return {}

  def derived(self) -> dict:
    """Values fixed at construction from something other than this component's
    own config -- an EnvStats entry, or a quantity the owning Policy computed and
    passed in (a critic's support, AMDP's erf_k). They are as load-bearing as any
    resolved kwarg and are not recoverable from config/ alone, so a Policy
    collects them into the run's logged config under `model/derived.*` rather
    than leaving them to be inferred."""
    return {}


class ActionSampler(Component, nnx.Module, ABC):
  """Canonical action generation, independent of a training objective."""

  critic_objective = "td"

  def __init__(self, env_stats=None, config=None, *, rngs: nnx.Rngs = None, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    self.rngs = rngs

  def next_key(self, key=None):
    """`key`, or a fresh draw from this sampler's own RNG stream."""
    return self.rngs.noise() if key is None else key

  @abstractmethod
  def generate_action(self, obs, *, critic: "Critic" = None, key=None) -> tuple:
    """(action, entropy): a canonical [-1, 1] action, one row per obs row, and
    the differential entropy of whatever distribution produced it, shaped `(B,)`.
    A sampler without an entropy estimate returns None for entropy."""

  def sample(self, obs, *, critic: "Critic" = None, key=None):
    """The played action alone -- the default just discards the entropy
    `generate_action` also returns. `Policy.inference_step` and the rollout
    loop call this, never `generate_action` directly."""
    return self.generate_action(obs, critic=critic, key=key)[0]


class Actor(ActionSampler):
  """A sampler with an off-policy actor objective."""

  @abstractmethod
  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    ...

class Critic(Component, nnx.Module, ABC):
  """A network that can price (s,a) -- pure architecture; Bellman targets
  and the regression loss itself live in each Policy's train_step."""

  @abstractmethod
  def value(self, s, a):
    """The only accessor an algorithm needs: a scalar price for (s, a). How that
    scalar is arrived at -- ensemble reduction, quantile mean, categorical
    expectation, pessimism -- is the implementation's business, never the
    caller's."""
    ...


