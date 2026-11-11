"""The declared return ranges, which are a distributional critic's fixed support.

`return_min`/`return_max` bound the **discounted value** a critic regresses, not
the undiscounted episode return -- at gamma 0.99 over a 1000-step horizon those
differ by 10x, and a support declared in the wrong units spends its atoms on
values the env cannot produce.

The convention is dime_am's (configs/alg/dime.yaml, and its README for the gym
envs): symmetric, at twice the discounted bound. These read class attributes
only; no env is built.
"""

import math

import jax.numpy as jnp
import pytest
from conftest import build, make_env_stats

from barl_jax.env import Environment


def declared(cls) -> dict:
    """The env class's `stats`, merged down the MRO as
    Environment.get_env_stats does it -- replicated rather than called so this
    stays free of a real gym env."""
    return cls.merged_stats()


@pytest.mark.parametrize("name", sorted(Environment.available()))
def test_every_support_is_symmetric(name):
    """The reference's convention throughout, and what keeps a near-uniform
    head's E[Z] -- and the pull of `entropy_coef` -- at 0 rather than off in the
    middle of a one-sided range."""
    stats = declared(Environment.class_for(name))
    assert stats["return_min"] == pytest.approx(-stats["return_max"])


@pytest.mark.parametrize("name", sorted(Environment.available()))
def test_every_environment_declares_a_usable_support(name):
    """A distributional critic defaults its support from these, so an absent or
    inverted range is a run that dies at construction."""
    stats = declared(Environment.class_for(name))
    lo, hi = stats.get("return_min"), stats.get("return_max")

    assert lo is not None and hi is not None, f"{name} declares no return range"
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < hi
    assert lo <= 0.0 <= hi, "the support has to contain zero -- untrained critics start there"


@pytest.mark.parametrize("q_gap", [0.2, 1.0])
def test_c51_resolves_a_realistic_dm_control_value_gap(q_gap):
    """The regression this file exists for. A terminal target at Q lands as a
    two-point split straddling it, so two targets a gap apart differ by exactly
    2*gap/bin_width in L1 -- the fraction of the learning signal that survives
    quantization. HumanoidRun's discounted value is ~0.2 under a random policy,
    a gap this test's `q_gap=0.2` case stands in for -- deliberately checked
    against whatever `bin_width` the current `num_bins`/support actually
    produces, not a hardcoded number: bin_width alone is a crude measure once
    the reward range isn't normalized, so this only asserts the quantization
    formula, never a specific config's tuned resolution."""
    stats = declared(Environment.class_for("HumanoidRun"))
    env_stats = make_env_stats(return_min=stats["return_min"], return_max=stats["return_max"])
    c = build("C51Critic", env_stats=env_stats)

    n = c.config.num_bins
    bin_width = (c.v_max - c.v_min) / (n - 1)
    boot = jnp.zeros((1, n)).at[0, n // 2].set(1.0)
    terminal = lambda q: c.bellman_target(jnp.array([q]), jnp.ones((1,)), jnp.ones((1,)), boot)

    # Start on an atom: with an even `num_bins` zero is a bin midpoint, and a
    # target there is already a 50/50 split, which the formula doesn't model.
    q0 = float(c.support_values()[n // 2])
    l1 = float(jnp.abs(terminal(q0) - terminal(q0 + q_gap)).sum())
    assert l1 == pytest.approx(min(2.0, 2 * q_gap / bin_width), rel=1e-3)
