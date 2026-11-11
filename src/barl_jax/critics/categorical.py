"""Distributional critic presets: `QCritic` with a categorical head.

The heads themselves (`C51`, `HLGauss`) and their supports live in `heads.py`
and `supports.py`; these names only pick their defaults, so
`{class: C51Critic, head: {support: {class: FixedSupport}}}` and
`{class: QCritic, head: {class: C51}}` build the same thing.
"""

from dataclasses import dataclass, field

from barl_jax.registration import register_component

from .base import QCritic, QCriticConfig
from .heads import (  # noqa: F401  (public surface)
  C51HeadConfig,
  CategoricalHeadConfig,
  HeadConfig,
  HLGaussHeadConfig,
  categorical_projection,
  to_histogram,
)


@dataclass(frozen=True)
class C51CriticConfig(QCriticConfig):
  class_name: str = "C51Critic"
  head: HeadConfig = field(default_factory=C51HeadConfig)


@dataclass(frozen=True)
class HLGaussCriticConfig(QCriticConfig):
  class_name: str = "HLGaussCritic"
  head: HeadConfig = field(default_factory=HLGaussHeadConfig)


@register_component
class C51Critic(QCritic):
  """Atoms on the support; the bootstrap distribution is projected."""

  config_type = C51CriticConfig


@register_component
class HLGaussCritic(QCritic):
  """Bin centers as support; the scalar TD target is smoothed into a histogram."""

  config_type = HLGaussCriticConfig
