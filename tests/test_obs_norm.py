"""Critic-level observation normalization (`QCritic.obs_norm`).

Raw DMC Dog observations reach |x| ~ 1e3 with a dozen dimensions of std 10-60.
A LayerNorm trunk without per-feature input normalization then barely sees the
action, the critic's Q goes flat in it, and an actor following dQ/da has
nothing to climb. Floq and Value Forcing use such a trunk, so they normalize
observations with running statistics by default.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import build, make_env_stats
from flax import nnx

from barl_jax.compat import upgrade_saved_model
from barl_jax.components import TRAINABLE
from barl_jax.critics.base import RunningObsNorm

BASE, NUISANCE, ACT = 4, 13, 3


def test_running_moments_pool_unequal_batches_exactly():
  norm = RunningObsNorm(3)
  x = jnp.asarray(np.random.default_rng(0).normal([1.0, -5.0, 100.0], [2.0, 0.1, 40.0], (50, 3)),
                  jnp.float32)
  np.testing.assert_allclose(norm(x[:2]), jnp.clip(x[:2], -10, 10), rtol=1e-4)  # ~identity before data
  norm.update(x[:7])
  norm.update(x[7:])
  np.testing.assert_allclose(norm.mean[...], x.mean(0), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(norm.var[...], x.var(0), rtol=1e-4)
  assert float(norm.count[...]) == 50
  z = norm(x)
  np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-3)
  np.testing.assert_allclose(z.std(0), 1.0, rtol=1e-3)
  assert float(jnp.abs(norm(jnp.full((1, 3), 1e6))).max()) == RunningObsNorm.CLIP


@pytest.mark.parametrize("name", ["FloqCritic", "ValueForcingCritic"])
def test_flow_critics_normalize_observations_by_default(name):
  critic = build(name, env_stats=make_env_stats())
  assert critic.config.obs_norm and critic.obs_norm is not None
  assert build("C51Critic").obs_norm is None, "the CrossQ trunk normalizes its own input"


def _dog_like_batch(key, n):
  """Observations with Dog-scale irrelevant features; reward depends on the action only."""
  k = jax.random.split(key, 3)
  obs = jnp.concatenate([jax.random.normal(k[0], (n, BASE)),
                         60.0 * jax.random.normal(k[1], (n, NUISANCE))], -1)
  action = jax.random.uniform(k[2], (n, ACT), minval=-1.0, maxval=1.0)
  return obs, action, 2.0 * action[:, 0]


@pytest.mark.parametrize("name, values", [
  ("FloqCritic", {"flow_steps": 2, "noise_samples": 2, "value_samples": 2}),
  ("ValueForcingCritic", {"flow_steps": 2, "return_scale": 1.0}),
])
def test_flow_critics_learn_the_action_through_large_observation_features(name, values):
  """Terminal rewards 2 * a[0] behind 13 observation features of std 60: the
  critic must learn dQ/da[0] = 2. Without `obs_norm` both flow critics stay
  flat in the action here (correlation ~0 after thousands of steps)."""
  stats = make_env_stats(obs_dim=BASE + NUISANCE, obs_shape=(BASE + NUISANCE,), act_dim=ACT,
                         act_low=np.full(ACT, -1.0, np.float32),
                         act_high=np.full(ACT, 1.0, np.float32))
  critic = build(name, env_stats=stats, latent_dim=64, layers=2, **values)
  optimizer = nnx.Optimizer(critic, critic.config.build(), wrt=TRAINABLE)
  n = 128

  @nnx.jit
  def update(c, opt, key):
    data_key, loss_key = jax.random.split(key)
    obs, action, reward = _dog_like_batch(data_key, n)

    def loss(c):
      return c.td_loss(obs, action, obs, action, reward, jnp.ones(n), jnp.ones(n, jnp.int32),
                       key=loss_key)["critic"]

    opt.update(c, nnx.grad(loss, argnums=nnx.DiffState(0, TRAINABLE))(c))
    c.after_step()

  for step in range(600):
    update(critic, optimizer, jax.random.fold_in(jax.random.key(0), step))

  obs, action, _ = _dog_like_batch(jax.random.key(99), 256)
  frozen = nnx.clone(critic)
  frozen.eval()
  gradient = nnx.grad(lambda c, a: c.value(obs, a).sum(), argnums=1)(frozen, action)
  assert float(gradient[:, 0].mean()) > 1.0
  assert float(jnp.abs(gradient[:, 1:]).mean()) < 0.5 * float(gradient[:, 0].mean())
  assert float(critic.obs_norm.count[...]) == 600 * n


@pytest.mark.parametrize("cls, expected", [
  ("FloqCritic", False), ("ValueForcingCritic", False), ("C51Critic", None),
])
def test_old_checkpoints_keep_the_normalization_they_were_trained_with(cls, expected):
  model = {"policy": "SAC", "components": {"critic": {"class": cls, "latent_dim": 8}}}
  old = upgrade_saved_model(model, 4)["components"]["critic"]
  assert old.get("obs_norm") is expected
  assert "obs_norm" not in upgrade_saved_model(model, 5)["components"]["critic"]
  explicit = {"policy": "SAC", "components": {"critic": {"class": cls, "obs_norm": True}}}
  assert upgrade_saved_model(explicit, 4)["components"]["critic"]["obs_norm"] is True
