"""CQL(H) as a critic axis: the penalty is the importance-weighted logsumexp gap, it
adds linearly to the head's own TD loss, and it needs a stat-free network."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import BATCH, HEADS, build, finite

PLAIN_MLP = {"class_name": "MLP", "activation": "relu", "norm": "none"}
N = 3


def cql_critic(name, **cql):
  return build(name, mlp=PLAIN_MLP, cql={"enabled": True, "n_actions": N, **cql})


def samples(c, key):
  a_dim = c.act_bounds.shape[-1]
  k = jax.random.split(key, 4)
  actions = lambda kk: jnp.tanh(jax.random.normal(kk, (N, BATCH, a_dim)))
  return actions(k[0]), jax.random.normal(k[1], (N, BATCH)), actions(k[2]), jax.random.normal(k[3], (N, BATCH))


@pytest.mark.parametrize("name", HEADS)
def test_penalty_is_the_importance_weighted_logsumexp_gap(name, obs, act, key):
  c = cql_critic(name, alpha=0.7, temperature=0.5)
  next_obs = obs[::-1]
  curr, curr_logp, nxt, nxt_logp = smp = samples(c, key)
  data = c.expectation_members(c.raw(obs, act))
  got = c.cql_penalty(obs, next_obs, data, smp, key)

  low, high = np.asarray(c.act_bounds[...])
  rand = np.asarray(jax.random.uniform(key, curr.shape, jnp.float32, low, high))
  rand_logp = np.full((N, BATCH), -np.log(high - low).sum())
  q = lambda s, a: np.asarray(c.expectation_members(c.raw(
    jnp.broadcast_to(s, (N,) + s.shape).reshape(N * BATCH, -1), a.reshape(N * BATCH, -1)))
  ).reshape(-1, N, BATCH)
  qs = np.concatenate([q(obs, rand), q(obs, curr), q(next_obs, nxt)], axis=1)
  logp = np.concatenate([rand_logp, np.asarray(curr_logp), np.asarray(nxt_logp)])
  t = 0.5
  lse = t * np.log(np.exp((qs - logp[None]) / t).sum(axis=1))
  want = 0.7 * (lse - np.asarray(data)[..., 0]).mean()
  assert float(got) == pytest.approx(want, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("name", HEADS)
def test_penalty_adds_linearly_to_the_critic_loss(name, obs, act, key):
  c = cql_critic(name)
  smp = samples(c, key)

  def loss(alpha):
    c.config = replace(c.config, cql=replace(c.config.cql, alpha=alpha))
    out = c.td_loss(obs, act, obs[::-1], act, jnp.zeros((BATCH,)), jnp.zeros((BATCH,)),
                    jnp.ones((BATCH,)), key=key, cql_samples=smp)
    assert set(out) == set(c.create_optimizer())
    return out["critic"]

  l0, l1, l2 = loss(0.0), loss(1.0), loss(2.0)
  assert finite(l2) and not jnp.allclose(l0, l1)
  assert float(l2 - l1) == pytest.approx(float(l1 - l0), rel=1e-3, abs=1e-4)


def test_cql_needs_samples_and_a_key(obs, act, key):
  c = cql_critic("DirectCritic")
  with pytest.raises(ValueError, match="cql_samples"):
    c.td_loss(obs, act, obs, act, jnp.zeros((BATCH,)), jnp.zeros((BATCH,)), jnp.ones((BATCH,)),
              key=key)


def test_cql_rejects_batch_statistic_norms():
  with pytest.raises(ValueError, match="cql"):
    build("DirectCritic", cql={"enabled": True})
