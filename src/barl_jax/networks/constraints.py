"""Parameter constraints: projections applied to a component's networks.

A constraint is selected per component with `constraints: [{class: WeightNorm}]`
and works on every component that owns a network -- critics and actors alike.
The owning Policy applies each one right after the component is built and again
after every optimizer step, before the component's own `after_step` (so a
target network blends towards the projected weights).

A constraint acts on every *trunk* inside the component: any module exposing
`hidden_linears` and `head` (MLP, SimBa, AdaLNSimBa), ensembles included, target
copies excluded. Adding a constraint is one registered `ConstraintConfig`
subclass with an `apply(trunk)`.
"""

from dataclasses import dataclass
from typing import ClassVar

from flax import nnx

from barl_jax.config.serialization import variant_registry
from barl_jax.networks.base import normalize_weights

CONSTRAINTS, register_constraint = variant_registry()


@dataclass(frozen=True)
class ConstraintConfig:
  variants: ClassVar[dict[str, type]] = CONSTRAINTS

  class_name: str = ""

  def apply(self, trunk) -> None:
    raise NotImplementedError

  def problems(self) -> list[str]:
    return []


@register_constraint
@dataclass(frozen=True)
class WeightNormConfig(ConstraintConfig):
  """XQC's unit-norm weight projection (arXiv:2509.25174): every hidden layer
  with its bias, the head's kernel only."""
  class_name: str = "WeightNorm"

  def apply(self, trunk) -> None:
    normalize_weights(trunk)


def trunks(module) -> list:
  """Every network inside `module` a constraint can project, skipping frozen
  target copies (their path carries the `_target` suffix) and trunks nested in
  another trunk."""
  from barl_jax.components import TARGET_SUFFIX
  candidates = {tuple(path): node for path, node in nnx.iter_graph(module)
                if _is_trunk(node) and not any(str(p).endswith(TARGET_SUFFIX) for p in path)}
  return [node for path, node in candidates.items()
          if not any(other != path and path[:len(other)] == other for other in candidates)]


def _is_trunk(node) -> bool:
  # Checked on the class: `nnx.Rngs` answers any attribute lookup with a stream.
  return (isinstance(node, nnx.Module)
          and isinstance(getattr(type(node), "hidden_linears", None), property)
          and hasattr(node, "head"))


def apply_constraints(component) -> None:
  for constraint in getattr(component.config, "constraints", ()) or ():
    for trunk in trunks(component):
      constraint.apply(trunk)
