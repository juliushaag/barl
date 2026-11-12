"""Named critic recipes, as data.

A preset is a partial critic config: `common` is merged into any critic, and
`categorical_head` additionally into a categorical head's config. Applying one
never needs to know which head or network the critic already has.

    critic = apply_preset(C51CriticConfig(), "xqc")
"""

from typing import Any, Mapping

from barl_jax.config.serialization import merge, to_dict

from .heads import CategoricalHeadConfig
from .mlps import target_mlp, xqc_mlp

PRESETS: dict[str, dict[str, Any]] = {
  # Polyak target net; batch-statistic norms need the joint forward.
  "target_ensemble": {
    "common": {"target": {"class": "Polyak", "joint_forward": True},
               "mlp": to_dict(target_mlp())},
  },
  # XQC (Palenicek et al., arXiv:2509.25174): polyak target sharing the joint
  # batch, pre-activation BatchNorm, weight normalization, min-member bootstrap.
  "xqc": {
    "common": {"target": {"class": "Polyak", "joint_forward": True},
               "ensemble": {"n_models": 2, "value_reduce": "mean", "target_reduce": "min"},
               "mlp": to_dict(xqc_mlp()), "constraints": [{"class": "WeightNorm"}],
               "latent_dim": 512, "layers": 3, "adam_b1": 0.9},
    "categorical_head": {"num_bins": 101, "entropy_coef": 0.0},
  },
  # NCLQL: noise-conditioned TD on a polyak/batch-norm critic.
  "nclql": {
    "common": {"target": {"class": "Polyak", "joint_forward": True},
               "mlp": to_dict(target_mlp()), "objective": {"class": "NoiseConditionedTD"}},
  },
}


def apply_preset(config, preset: str | Mapping[str, Any]):
  """`config` with a named (or literal) preset merged in."""
  spec = PRESETS[preset] if isinstance(preset, str) else preset
  config = merge(config, spec.get("common", {}))
  if isinstance(config.head, CategoricalHeadConfig) and spec.get("categorical_head"):
    from dataclasses import replace
    config = replace(config, head=merge(config.head, spec["categorical_head"], path="head"))
  return config


# Function forms, for call sites that read better as `xqc(config)`.
def target_ensemble(config):
  return apply_preset(config, "target_ensemble")


def xqc(config):
  return apply_preset(config, "xqc")


def nclql(config):
  return apply_preset(config, "nclql")
