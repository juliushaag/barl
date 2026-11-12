"""Scalar-head critic presets: named defaults for `QCritic`, nothing more."""

from dataclasses import dataclass, field

from barl_jax.registration import register_component

from .base import EnsembleConfig, QCritic, QCriticConfig
from .heads import HeadConfig, ScalarHeadConfig


@dataclass(frozen=True)
class DirectCriticConfig(QCriticConfig):
  class_name: str = "DirectCritic"
  head: HeadConfig = field(default_factory=ScalarHeadConfig)
  ensemble: EnsembleConfig = field(default_factory=lambda: EnsembleConfig(n_models=1))


@dataclass(frozen=True)
class EnsembleCriticConfig(DirectCriticConfig):
  class_name: str = "EnsembleCritic"
  ensemble: EnsembleConfig = field(default_factory=lambda: EnsembleConfig(
    n_models=2, value_reduce="min", target_reduce="min"))


@register_component
class DirectCritic(QCritic):
  """Scalar Q regressed with MSE, one member."""

  config_type = DirectCriticConfig


@register_component
class EnsembleCritic(QCritic):
  """Two scalar members, min-reduced (clipped double Q)."""

  config_type = EnsembleCriticConfig
