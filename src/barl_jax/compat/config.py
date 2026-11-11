"""Upgrades for component config mappings written against an older schema.

`upgrade_component` runs on every component mapping before strict construction:
experiment YAML, `--set` / grid overrides (partial mappings), and the model spec
saved inside a checkpoint (complete mappings). Each rule rewrites only keys that
no longer exist, so a current mapping passes through unchanged, and a partial
override is rewritten key by key.

Rules, oldest first. Remove a rule once nothing that predates it needs loading.
"""

from typing import Any, Callable, Mapping

# The critic classes whose config used the pre-parts flat layout.
_CRITICS = {"QCritic", "DirectCritic", "EnsembleCritic", "C51Critic", "HLGaussCritic",
            "FloqCritic"}
_CATEGORICAL = {"C51Critic", "HLGaussCritic"}


def _weight_norm_to_constraints(values: dict, class_name: str | None) -> dict:
  """policy format 3 -> 4: `weight_norm: bool` became the generic
  `constraints: [{class: WeightNorm}]`, available to every component."""
  if "weight_norm" not in values:
    return values
  enabled = values.pop("weight_norm")
  if enabled:
    constraints = list(values.get("constraints", ()))
    if not any(_class_of(c) == "WeightNorm" for c in constraints):
      constraints.append({"class_name": "WeightNorm"})
    values["constraints"] = constraints
  return values


def _target_kind_to_part(values: dict, class_name: str | None) -> dict:
  """policy format 3 -> 4: `target: {kind: none|polyak, ...}` became a part
  selected by class, `NoTarget` (no tau/delay) or `Polyak`."""
  target = values.get("target")
  if not isinstance(target, Mapping) or "kind" not in target:
    return values
  target = dict(target)
  kind = target.pop("kind")
  if kind == "polyak":
    target["class_name"] = "Polyak"
  else:
    target["class_name"] = "NoTarget"
    target.pop("tau", None)
    target.pop("delay", None)
  values["target"] = target
  return values


def _noise_to_objective(values: dict, class_name: str | None) -> dict:
  """policy format 3 -> 4: `noise: {enabled, schedule, time, smoothing_coef}`
  became the `objective` part, `TD` or `NoiseConditionedTD`."""
  if "noise" not in values:
    return values
  noise = dict(values.pop("noise"))
  enabled = noise.pop("enabled", None)
  if enabled is False:
    values["objective"] = {"class_name": "TD"}
  elif enabled is True:
    values["objective"] = {"class_name": "NoiseConditionedTD", **noise}
  elif noise:     # a partial override of the noise settings alone
    values["objective"] = noise
  return values


_HEAD_FIELDS = ("num_bins", "entropy_coef", "sigma_to_bin_ratio")
_LEARNED_FIELDS = {"support_init_min": "init_min", "support_init_max": "init_max",
                   "support_alpha": "alpha", "support_eps": "eps",
                   "support_sigma_min": "sigma_min", "support_lr": "lr",
                   "dual_lr": "dual_lr", "dual_init": "dual_init"}
_FIXED_FIELDS = ("v_min", "v_max")


def _categorical_fields_to_head(values: dict, class_name: str | None) -> dict:
  """policy format 3 -> 4: a categorical critic's flat fields moved into its
  `head`, and DySEL's (`learn_support`, `support_*`, `dual_*`) into
  `head.support`, a `LearnedSupport` or a `FixedSupport` (with `v_min/v_max`)."""
  if class_name not in _CATEGORICAL:
    return values
  head = dict(values.get("head", {}))
  for name in _HEAD_FIELDS:
    if name in values:
      head[name] = values.pop(name)
  support = dict(head.get("support", {}))
  learn = values.pop("learn_support", None)
  learned = {new: values.pop(old) for old, new in _LEARNED_FIELDS.items() if old in values}
  fixed = {name: values.pop(name) for name in _FIXED_FIELDS if name in values}
  if learn is True:
    support.update(class_name="LearnedSupport", **learned)
  elif learn is False:
    support.update(class_name="FixedSupport", **fixed)
  else:           # a partial override that does not switch the support
    support.update(learned if learned else fixed)
  if support:
    head["support"] = support
  if head:
    values["head"] = head
  return values


RULES: tuple[Callable[[dict, str | None], dict], ...] = (
  _weight_norm_to_constraints,
  _target_kind_to_part,
  _noise_to_objective,
  _categorical_fields_to_head,
)


def _class_of(values) -> str | None:
  if isinstance(values, Mapping):
    return values.get("class_name", values.get("class"))
  return getattr(values, "class_name", None)


def upgrade_component(values: Mapping[str, Any], class_name: str | None = None) -> dict:
  """`values` (one component's mapping, complete or partial) in the current
  schema. `class_name` is the component's class when the mapping itself does
  not name one (a partial override of an existing slot)."""
  values = dict(values)
  class_name = _class_of(values) or class_name
  if class_name not in _CRITICS and "weight_norm" not in values:
    return values
  for rule in RULES:
    values = rule(values, class_name)
  return values
