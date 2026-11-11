"""Backward compatibility (`barl_jax.compat`): old checkpoints and old configs.

`fixtures/policy_v3/*` were written by the release before critics became parts,
each with every float leaf perturbed (so a fresh init can never pass for a
restore) and with the Q values, a sampled action and one train step's losses
recorded next to it. Loading one with the current code must reproduce them.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from conftest import make_env_stats
from flax import nnx
from test_policies import FakeEnv, batch, make

from barl_jax.checkpoints import load_policy, save_policy, saved_model
from barl_jax.compat import POLICY_VERSION, check_version, restore_state, upgrade_component
from barl_jax.compat.checkpoint import saved_path
from barl_jax.components import TRAINABLE
from barl_jax.config.run import RunConfig
from barl_jax.config.serialization import to_dict
from barl_jax.optimization import train_step

FIXTURES = Path(__file__).parent / "fixtures" / "policy_v3"
V3 = sorted(p.name for p in FIXTURES.iterdir() if (p / "meta.json").exists())


def outputs(policy) -> dict:
  data = batch()
  obs, act = data["obs"], data["act"]
  out = {"q": np.asarray(policy.critic.value(obs, act.reshape(act.shape[0], -1))),
         "action": np.asarray(policy.actor.sample(obs, critic=policy.critic,
                                                  key=jax.random.key(7)))}
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  graphdef, state = nnx.split((policy, optimizer))
  _, losses, _, _ = train_step(graphdef, state, data, (), jnp.asarray(0, jnp.int32))
  out.update({f"loss/{k}": np.asarray(v) for k, v in losses.items()})
  return out


@pytest.mark.parametrize("name", V3)
def test_v3_checkpoint_loads_and_reproduces_its_outputs(name):
  policy = load_policy(FIXTURES / name, FakeEnv(), make_env_stats())
  reference = dict(np.load(FIXTURES / name / "reference.npz"))
  got = outputs(policy)
  assert set(got) == set(reference)
  for key, want in reference.items():
    np.testing.assert_allclose(got[key], want, rtol=1e-5, atol=1e-6, err_msg=key)


def test_saving_writes_the_current_version(tmp_path):
  policy = make("SAC")
  save_policy(policy, tmp_path / "policy")
  _, meta = saved_model(tmp_path / "policy")
  assert meta["format_version"] == POLICY_VERSION


@pytest.mark.parametrize("version", [2, POLICY_VERSION + 1, None, "3"])
def test_unreadable_versions_are_rejected_with_the_supported_range(version):
  with pytest.raises(ValueError, match="compat/checkpoint.py"):
    check_version(version)


def test_saved_paths_invert_the_renames_under_any_prefix():
  """The same rule maps policy state and an optimizer's moments, which nest the
  critic's parameters under their own prefix."""
  assert saved_path(("critic", "head", "support", "endpoints"), 3) == ("critic", "support_net",
                                                                      "endpoints")
  assert saved_path(("critic", "head", "support", "centers"), 3) == ("critic", "support")
  assert saved_path(("critic", "target", "net_target", "x"), 3) == ("critic", "net_target", "x")
  assert saved_path(("opt_state", "0", "critic", "head", "support", "dual"), 3) == \
    ("opt_state", "0", "critic", "dual")
  assert saved_path(("actor", "head", "support"), 3) == ("actor", "head", "support")
  assert saved_path(("critic", "head", "support", "dual"), POLICY_VERSION) == \
    ("critic", "head", "support", "dual")


def test_optimizer_state_restores_through_the_renames(tmp_path):
  """A resume checkpoint stores `(policy, optimizer)` together. Write one in the
  v3 layout by hand and restore it into the current one."""
  policy = make("SAC")
  optimizer = nnx.Optimizer(policy, policy.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  state = nnx.state((policy, optimizer))
  flat = list(nnx.to_flat_state(state))
  key = jax.random.key(3)
  old = {}
  expected = {}
  for path, variable in flat:
    value = variable[...]
    if jnp.issubdtype(value.dtype, jnp.floating):
      key, sub = jax.random.split(key)
      value = value + jax.random.normal(sub, value.shape, value.dtype)
    old[saved_path(path, 3) + ("value",)] = value
    expected[path] = value
  assert any(p[0] == "1" and "support_net" in p for p in old), \
    "no optimizer moment sits under a renamed path"
  with ocp.PyTreeCheckpointer() as ckpt:
    ckpt.save(tmp_path / "state", nnx.traversals.unflatten_mapping(old))

  fresh = make("SAC")
  fresh_optimizer = nnx.Optimizer(fresh, fresh.build_optimizer(lr_schedule=None), wrt=TRAINABLE)
  restored = restore_state(tmp_path / "state", nnx.state((fresh, fresh_optimizer)), 3)
  raw = lambda x: jax.random.key_data(x) if jnp.issubdtype(x.dtype, jax.dtypes.prng_key) else x
  for path, variable in nnx.to_flat_state(restored):
    np.testing.assert_array_equal(raw(variable[...]), raw(expected[path]), err_msg=str(path))


# ---- configs -------------------------------------------------------------------

OLD_C51 = {
  "class": "C51Critic", "lr": 3e-4, "adam_b1": 0.5, "grad_clip": 1.0,
  "grad_clip_mode": "global_norm", "gamma": 0.99, "reward_scale": 1.0, "latent_dim": 64,
  "layers": 2, "weight_norm": True,
  "target": {"kind": "polyak", "tau": 0.01, "delay": 2, "joint_forward": True},
  "ensemble": {"n_models": 2, "value_reduce": "mean", "target_reduce": "min"},
  "noise": {"enabled": False, "smoothing_coef": 1.0},
  "num_bins": 51, "v_min": None, "v_max": None, "entropy_coef": 0.0,
  "sigma_to_bin_ratio": 0.75, "learn_support": True, "support_init_min": -5.0,
  "support_init_max": 5.0, "support_alpha": 0.3, "support_eps": 0.005,
  "support_sigma_min": 0.3, "support_lr": 0.003, "dual_lr": 0.001, "dual_init": 1.0,
}


def test_a_complete_old_critic_mapping_upgrades_field_by_field():
  new = upgrade_component(OLD_C51)
  assert new["constraints"] == [{"class_name": "WeightNorm"}]
  assert new["target"] == {"class_name": "Polyak", "tau": 0.01, "delay": 2, "joint_forward": True}
  assert new["objective"] == {"class_name": "TD"}
  assert new["head"]["num_bins"] == 51 and new["head"]["entropy_coef"] == 0.0
  assert new["head"]["support"] == {"class_name": "LearnedSupport", "init_min": -5.0,
                                    "init_max": 5.0, "alpha": 0.3, "eps": 0.005,
                                    "sigma_min": 0.3, "lr": 0.003, "dual_lr": 0.001,
                                    "dual_init": 1.0}
  for gone in ("weight_norm", "noise", "learn_support", "num_bins", "v_min", "support_alpha"):
    assert gone not in new


def test_current_mappings_pass_through_unchanged():
  current = to_dict(RunConfig.from_dict({"model": "SAC", "env": "HalfCheetah"})
                    .model_config.components.critic)
  assert upgrade_component(current) == current


@pytest.mark.parametrize("old, new", [
  ({"learn_support": False}, {"head": {"support": {"class": "FixedSupport"}}}),
  ({"learn_support": False, "v_min": -3.0, "v_max": 3.0},
   {"head": {"support": {"class": "FixedSupport", "v_min": -3.0, "v_max": 3.0}}}),
  ({"support_alpha": 0.5}, {"head": {"support": {"alpha": 0.5}}}),
  ({"entropy_coef": 0.0}, {"head": {"entropy_coef": 0.0}}),
  ({"weight_norm": True}, {"constraints": [{"class": "WeightNorm"}]}),
  ({"target": {"kind": "polyak", "tau": 0.02}}, {"target": {"class": "Polyak", "tau": 0.02}}),
  ({"noise": {"smoothing_coef": 0.5}}, {"objective": {"smoothing_coef": 0.5}}),
])
def test_old_overrides_build_the_same_model_as_new_ones(old, new):
  model = "NCLQL" if "noise" in old else "SAC"
  build = lambda critic: RunConfig.from_dict({
    "model": model, "env": "HalfCheetah",
    "model_overrides": {"components": {"critic": critic}}}).model_config
  assert build(old) == build(new)
