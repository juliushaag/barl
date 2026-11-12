"""The Value Forcing critic (arXiv:2611.02160): a return flow trained with the
flow Bellman backup at every noise level.

The generic critic suite covers the shared interface. These check what is
specific to it: the backup equation itself, that it learns values, and that it
learns a return *distribution* rather than collapsing to its mean.
"""

import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import BATCH, OBS_DIM, build, make_env_stats
from flax import nnx

from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig
from barl_jax.critics.value_forcing import gaussian_quantiles
from barl_jax.metrics import DEBUG, PRIMARY
from barl_jax.optimization import train_step
from barl_jax.policy import Policy

ACT_DIM = 3


class _BoxEnv:
  observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
  action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)


def _critic(**values):
  return build("ValueForcingCritic", env_stats=make_env_stats(), **values)


def _batch():
  keys = jax.random.split(jax.random.key(90), 5)
  return {
    "obs": jax.random.normal(keys[0], (BATCH, OBS_DIM)),
    "next_obs": jax.random.normal(keys[1], (BATCH, OBS_DIM)),
    "act": jnp.tanh(jax.random.normal(keys[2], (BATCH, ACT_DIM))),
    "next_act": jnp.tanh(jax.random.normal(keys[3], (BATCH, ACT_DIM))),
    "chunk_return": jax.random.normal(keys[4], (BATCH,)),
    "term": jnp.asarray([0.0, 1.0] * (BATCH // 2)),
    "horizon": jnp.asarray([1, 3] * (BATCH // 2), jnp.int32),
  }


class _Affine(nnx.Module):
  """A velocity with a closed form, `a * z + b * tau + c * action[0]`, the same
  for every member, so the backup can be recomputed by hand."""

  def __init__(self, a, b, c, n_models):
    self.a, self.b, self.c, self.n = a, b, c, n_models

  def __call__(self, obs, action, z, tau):
    v = self.a * z + self.b * tau + self.c * action[:, :1]
    return jnp.broadcast_to(v, (self.n, *v.shape[-2:]))


def test_value_is_finite_deterministic_and_jittable():
  critic = _critic()
  data = _batch()
  eager = critic.value(data["obs"], data["act"])
  assert eager.shape == (BATCH, 1) and bool(jnp.all(jnp.isfinite(eager)))
  np.testing.assert_array_equal(eager, critic.value(data["obs"], data["act"]))

  graphdef, state = nnx.split(critic)
  compiled = jax.jit(lambda g, s, o, a: nnx.merge(g, s).value(o, a))(
    graphdef, state, data["obs"], data["act"])
  np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-5)


def test_gaussian_quantiles_are_a_standard_normal_sample():
  z = gaussian_quantiles(64)
  assert float(jnp.abs(z.mean())) < 1e-5
  assert float(z.std()) == pytest.approx(1.0, abs=0.05)
  np.testing.assert_allclose(z, -z[::-1], atol=1e-5)


@pytest.mark.parametrize("bonus", [0.0, 0.7])
def test_loss_is_the_flow_bellman_backup(bonus):
  """Recompute their Eq. 13, generalised to n-step returns and termination,
  from the same random draws `td_loss` makes."""
  critic = _critic(flow_steps=4)
  live = _Affine(0.3, -0.2, 0.5, critic.n_models)
  target_net = _Affine(-0.4, 0.1, 0.25, critic.n_models)
  critic.net, critic.target.net_target = live, target_net
  data = _batch()
  key = jax.random.key(11)
  entropy_bonus = jnp.full((BATCH, 1), bonus)

  loss = critic.td_loss(data["obs"], data["act"], data["next_obs"], data["next_act"],
                        data["chunk_return"], data["term"], data["horizon"],
                        entropy_bonus=entropy_bonus, key=key)["critic"]

  tau_key, start_key, eps_key = jax.random.split(key, 3)
  tau = jax.random.uniform(tau_key, (BATCH, 1))
  z = jax.random.normal(start_key, (BATCH, 1))
  eps = jax.random.normal(eps_key, (BATCH, 1))
  step = tau / 4
  for i in range(4):                                  # Euler to tau under the target flow
    z = z + step * (-0.4 * z + 0.1 * (i * step) + 0.25 * data["next_act"][:, :1])
  v_next = -0.4 * z + 0.1 * tau + 0.25 * data["next_act"][:, :1]

  gamma, scale = critic.config.gamma, 1.0 - critic.config.gamma
  g = (1.0 - data["term"][:, None]) * gamma ** data["horizon"][:, None]
  drift = scale * (data["chunk_return"][:, None] + g * entropy_bonus)
  spread = jnp.sqrt(1.0 - g ** 2)
  z_in = tau * drift + g * z + (1.0 - tau) * spread * eps
  target = drift + g * v_next - spread * eps
  pred = 0.3 * z_in - 0.2 * tau + 0.5 * data["act"][:, :1]
  expected = critic.n_models * ((pred - target) ** 2).mean()
  np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_both_sides_of_the_backup_are_standard_normal_at_noise_level_zero():
  """The Gaussian correction's job: at tau = 0 the input the critic is trained
  on is N(0, 1) whatever the discount, so the flow starts from the same noise
  it is sampled from."""
  g = jnp.asarray([0.0, 0.5, 0.99])[:, None]
  z0 = jax.random.normal(jax.random.key(0), (1, 200_000))
  eps = jax.random.normal(jax.random.key(1), (1, 200_000))
  z_in = g * z0 + jnp.sqrt(1.0 - g ** 2) * eps
  np.testing.assert_allclose(z_in.mean(axis=1), 0.0, atol=0.01)
  np.testing.assert_allclose(z_in.std(axis=1), 1.0, atol=0.01)


def test_loss_trains_the_live_velocity_only():
  critic = _critic(flow_steps=2)
  data = _batch()

  def loss_fn(c):
    return c.td_loss(data["obs"], data["act"], data["next_obs"], data["next_act"],
                     data["chunk_return"], data["term"], data["horizon"],
                     key=jax.random.key(3))["critic"]

  grads = nnx.state(nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(critic), nnx.Param)
  assert any(bool(jnp.any(g != 0)) for g in jax.tree.leaves(grads["net"]))
  assert all(bool(jnp.all(g == 0)) for g in jax.tree.leaves(grads["target"]["net_target"]))


def _train(critic, obs, action, reward, steps, terminated=None):
  n = obs.shape[0]
  terminated = jnp.ones(n) if terminated is None else terminated
  optimizer = nnx.Optimizer(critic, critic.config.build(), wrt=TRAINABLE)

  @nnx.jit
  def update(c, opt, key):
    def loss(c):
      return c.td_loss(obs, action, obs, action, reward, terminated, jnp.ones(n, jnp.int32),
                       key=key)["critic"]
    opt.update(c, nnx.grad(loss, argnums=nnx.DiffState(0, TRAINABLE))(c))
    c.after_step()

  for step in range(steps):
    update(critic, optimizer, jax.random.fold_in(jax.random.key(1), step))


def test_learns_action_dependent_terminal_rewards():
  """A one-step MDP has Q(s, a) = r(s, a) = 2 * a[0]: the value, and its action
  gradient (what an actor climbs), must both be learned."""
  critic = _critic(latent_dim=64, layers=2, flow_steps=2, return_scale=1.0)
  obs = jnp.zeros((64, OBS_DIM))
  action = jax.random.uniform(jax.random.key(52), (64, ACT_DIM), minval=-1.0, maxval=1.0)
  test_action = jax.random.uniform(jax.random.key(53), (64, ACT_DIM), minval=-1.0, maxval=1.0)
  target = 2 * test_action[:, 0]

  initial = jnp.mean((critic.value(obs, test_action).reshape(-1) - target) ** 2)
  _train(critic, obs, action, 2 * action[:, 0], steps=400)
  mse = jnp.mean((critic.value(obs, test_action).reshape(-1) - target) ** 2)
  assert float(mse) < 0.25 and float(mse) < 0.2 * float(initial)
  gradient = nnx.grad(lambda c, a: c.value(obs, a).sum(), argnums=1)(critic, test_action)
  assert float(gradient[:, 0].mean()) > 0.75


def test_bootstrapping_converges_to_the_discounted_return():
  """A self-loop paying reward 1 forever has Q = 1 / (1 - gamma) = 10 at
  gamma = 0.9. Nothing in the data states that number: it is only reachable by
  bootstrapping through the target flow at every noise level."""
  critic = _critic(latent_dim=64, layers=2, flow_steps=4, gamma=0.9,
                   target={"class": "Polyak", "tau": 0.05, "joint_forward": False})
  n = 128
  obs, action = jnp.zeros((n, OBS_DIM)), jnp.zeros((n, ACT_DIM))
  _train(critic, obs, action, jnp.ones(n), steps=800, terminated=jnp.zeros(n))

  assert float(critic.value(obs[:1], action[:1])[0, 0]) == pytest.approx(10.0, abs=1.5)
  samples = critic.return_samples(obs[:1], action[:1], jax.random.key(0), samples=256, steps=16)
  assert float(samples.mean()) == pytest.approx(10.0, abs=1.5)


def test_learns_the_return_distribution_not_just_its_mean():
  """Terminal rewards of -1 or +1 at random, independent of (s, a): the mean
  return is 0 and a collapsed critic would predict only that. The flow has to
  put its samples near both outcomes."""
  critic = _critic(latent_dim=64, layers=2, flow_steps=8, return_scale=1.0)
  n = 256
  obs = jnp.zeros((n, OBS_DIM))
  action = jnp.zeros((n, ACT_DIM))
  reward = jnp.where(jax.random.bernoulli(jax.random.key(4), 0.5, (n,)), 1.0, -1.0)
  _train(critic, obs, action, reward, steps=600)

  samples = critic.return_samples(obs[:1], action[:1], jax.random.key(9), samples=512, steps=16)
  samples = np.asarray(samples).reshape(-1)
  assert abs(samples.mean()) < 0.25
  assert samples.std() > 0.7, "the return distribution collapsed towards its mean"
  assert np.mean(np.abs(np.abs(samples) - 1.0) < 0.5) > 0.7, "samples are not near -1 / +1"


class _Split(nnx.Module):
  """Member 0's velocity is +1 and member 1's is -1, everywhere."""

  def __call__(self, obs, action, z, tau):
    shape = z.shape[-2:]
    return jnp.stack([jnp.ones(shape), -jnp.ones(shape)])


class _Echo(nnx.Module):
  """Predicts its own noisy-return input, per member."""

  def __call__(self, obs, action, z, tau):
    return jnp.broadcast_to(z, (2, *z.shape[-2:]))


@pytest.mark.parametrize("reduce", ["min", "mean"])
def test_target_reduction_selects_the_bootstrap_member(reduce):
  """`min`: every member bootstraps from the target member with the lower value
  estimate (member 1 here). `mean`: each from its own target copy."""
  critic = _critic(flow_steps=1, ensemble={"target_reduce": reduce})
  critic.target.net_target, critic.net = _Split(), _Echo()
  data = _batch()
  key = jax.random.key(5)
  loss = critic.td_loss(data["obs"], data["act"], data["next_obs"], data["next_act"],
                        jnp.zeros((BATCH,)), jnp.zeros((BATCH,)), jnp.ones((BATCH,), jnp.int32),
                        key=key)["critic"]

  tau_key, start_key, eps_key = jax.random.split(key, 3)
  tau = jax.random.uniform(tau_key, (BATCH, 1))
  z0 = jax.random.normal(start_key, (BATCH, 1))
  eps = jax.random.normal(eps_key, (BATCH, 1))
  g = critic.config.gamma
  spread = math.sqrt(1.0 - g ** 2)

  def member_loss(velocity):          # one Euler step to tau under a constant velocity
    z_in = g * (z0 + tau * velocity) + (1.0 - tau) * spread * eps
    return ((z_in - (g * velocity - spread * eps)) ** 2).mean()

  expected = 2 * member_loss(-1.0) if reduce == "min" else member_loss(1.0) + member_loss(-1.0)
  np.testing.assert_allclose(loss, expected, rtol=1e-5)


@pytest.mark.parametrize("values, match", [
  ({"flow_steps": 0}, "flow_steps"),
  ({"value_samples": 0}, "value_samples"),
  ({"return_scale": 0.0}, "return_scale"),
  ({"head": {"class": "C51"}}, "Scalar head"),
  ({"objective": {"class": "NoiseConditionedTD"}}, "TD only"),
  ({"mlp": {"norm": "batchrenorm", "norm_kwargs": {
    "momentum": 0.99, "epsilon": 0.001, "r_max": 3.0, "d_max": 5.0, "warmup_steps": 10}}},
   "batch statistics"),
])
def test_invalid_configurations_are_rejected(values, match):
  with pytest.raises(ValueError, match=match):
    _critic(**values)


@pytest.mark.parametrize("model", ["SAC", "TD3", "AMDP"])
def test_policies_take_a_training_step_with_the_critic(model):
  run = RunConfig.from_dict({
    "model": model, "env": "HalfCheetah",
    "model_overrides": {"components": {
      "actor": {"latent_dim": 8},
      "critic": {"class": "ValueForcingCritic", "latent_dim": 8, "flow_steps": 2}}},
  })
  policy = Policy.make(model, _BoxEnv(), run.model_config, env_stats=make_env_stats(), seed=0)
  assert type(policy.critic).__name__ == "ValueForcingCritic"
  assert policy.derived_config()["critic.return_scale"] == pytest.approx(0.01)

  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))
  data = {k: v for k, v in _batch().items() if k != "next_act"}
  state, losses, (primary, _), _ = train_step(graphdef, state, data, (PRIMARY, DEBUG),
                                              jnp.asarray(0))
  assert all(bool(jnp.all(jnp.isfinite(v))) for v in losses.values())
  assert "critic/q/mean" in primary and "critic/velocity/mean" in primary
