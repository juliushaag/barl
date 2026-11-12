"""Critics: `QCritic` and its parts. Imported for @register_component side
effects and as the public critic surface."""
from barl_jax.critics.base import (
  COMBINATION_RULES,
  EnsembleConfig,
  QCritic,
  QCriticConfig,
  QNetwork,
  n_step_discount,
  reduce_members,
)
from barl_jax.critics.categorical import (
  C51Critic,
  C51CriticConfig,
  HLGaussCritic,
  HLGaussCriticConfig,
)
from barl_jax.critics.direct import (
  DirectCritic,
  DirectCriticConfig,
  EnsembleCritic,
  EnsembleCriticConfig,
)
from barl_jax.critics.floq import FloqCritic, FloqCriticConfig
from barl_jax.critics.heads import (
  C51HeadConfig,
  CategoricalHeadConfig,
  HeadConfig,
  HLGaussHeadConfig,
  ScalarHeadConfig,
  categorical_projection,
  scalar_td_target,
  to_histogram,
)
from barl_jax.critics.mlps import crossq_mlp, target_mlp, xqc_mlp
from barl_jax.critics.objectives import NoiseConditionedTDConfig, ObjectiveConfig, TDConfig
from barl_jax.critics.presets import PRESETS, apply_preset
from barl_jax.critics.supports import (
  FixedSupportConfig,
  LearnedSupportConfig,
  SupportConfig,
  value_range,
)
from barl_jax.critics.targets import NoTargetConfig, PolyakConfig, TargetConfig
from barl_jax.critics.value_forcing import ValueForcingCritic, ValueForcingCriticConfig

__all__ = [
  "COMBINATION_RULES", "PRESETS",
  "C51Critic", "C51CriticConfig", "C51HeadConfig", "CategoricalHeadConfig",
  "DirectCritic", "DirectCriticConfig", "EnsembleConfig", "EnsembleCritic",
  "EnsembleCriticConfig", "FixedSupportConfig", "FloqCritic", "FloqCriticConfig",
  "HLGaussCritic", "HLGaussCriticConfig", "HLGaussHeadConfig", "HeadConfig",
  "LearnedSupportConfig", "NoTargetConfig", "NoiseConditionedTDConfig", "ObjectiveConfig",
  "PolyakConfig", "QCritic", "QCriticConfig", "QNetwork", "ScalarHeadConfig",
  "SupportConfig", "TDConfig", "TargetConfig", "apply_preset", "categorical_projection",
  "crossq_mlp", "n_step_discount", "reduce_members", "scalar_td_target", "target_mlp",
  "to_histogram", "value_range", "xqc_mlp", "ValueForcingCritic", "ValueForcingCriticConfig",
]
