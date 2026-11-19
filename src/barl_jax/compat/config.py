"""Upgrades for component config mappings written against an older schema.

`upgrade_component` runs on every component mapping before strict construction:
experiment YAML, `--set` / grid overrides (partial mappings), and the model spec
saved inside a checkpoint (complete mappings). Each rule rewrites only keys that
no longer exist, so a current mapping passes through unchanged, and a partial
override is rewritten key by key.

Rules, oldest first. Remove a rule once nothing that predates it needs loading.

`upgrade_model` / `upgrade_model_overrides` hold the rules that span the whole
model mapping (a policy param that became a field of every component).
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


def _cql_to_objective(values: dict, class_name: str | None) -> dict:
  """`cql: {enabled, alpha, n_actions, temperature}` (a local pre-parts field)
  became the `CQL` objective."""
  if "cql" not in values:
    return values
  cql = dict(values.pop("cql"))
  enabled = cql.pop("enabled", None)
  if enabled is True:
    values["objective"] = {"class_name": "CQL", **cql}
  elif enabled is None and cql:     # a partial override of the CQL settings alone
    values["objective"] = cql
  return values


_HEAD_FIELDS = ("num_bins", "entropy_coef", "sigma_to_bin_ratio")
_LEARNED_FIELDS = {"support_init_min": "init_min", "support_init_max": "init_max",
                   "support_alpha": "alpha", "support_eps": "eps",
                   "support_sigma_min": "sigma_min", "support_lr": "lr",
                   "dual_lr": "dual_lr", "dual_init": "dual_init",
                   "support_from_random": "from_random"}
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
  _cql_to_objective,
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


# ---- whole-model rules -------------------------------------------------------

def _has_input_norm(values) -> bool:
  from dataclasses import fields

  from barl_jax.registration import get_component
  try:
    config_type = get_component(_class_of(values)).config_type
  except (KeyError, TypeError):
    return False
  return config_type is not None and any(f.name == "input_norm" for f in fields(config_type))


def upgrade_model(values: Mapping[str, Any]) -> dict:
  """A complete saved model mapping in the current schema.

  policy format 4 -> 5: `params.normalize_obs` (one set of statistics owned by
  the Policy) became the `input_norm` part of every actor and critic. A model
  trained with it gets `RSNorm` without clipping and with the old std floor --
  the same mapping, and the state migration hands it the saved statistics; a
  model trained without it gets `Identity`, so it reproduces unchanged."""
  values = dict(values)
  params = dict(values.get("params", {}))
  if "normalize_obs" not in params:
    return values
  enabled = params.pop("normalize_obs")
  norm = ({"class_name": "RSNorm", "eps": 1.0e-12, "clip": None} if enabled
          else {"class_name": "Identity"})
  components = {}
  for slot, component in dict(values.get("components", {})).items():
    if isinstance(component, Mapping) and "input_norm" not in component \
        and _has_input_norm(component):
      component = {**component, "input_norm": dict(norm)}
    components[slot] = component
  values["params"] = params
  values["components"] = components
  return values


def upgrade_model_overrides(overrides: Mapping[str, Any], *, strict: bool = True) -> dict:
  """A partial `model_overrides` mapping (YAML, `--set`) in the current schema.

  policy format 4 -> 5: `params.normalize_obs: true` is now the default of every
  actor and critic (`input_norm: {class: RSNorm}`) and is dropped; `false` is
  refused, because a new run always normalizes its inputs per feature (lenient
  parsing of a saved run config warns and drops it instead)."""
  overrides = dict(overrides)
  params = overrides.get("params")
  if not isinstance(params, Mapping) or "normalize_obs" not in params:
    return overrides
  params = dict(params)
  if not params.pop("normalize_obs"):
    message = ("model_overrides.params.normalize_obs=false: observation normalization moved into "
               "each actor and critic (`input_norm`, RSNorm by default) and cannot be turned off "
               "for a new run; drop the key")
    if strict:
      raise ValueError(message)
    import warnings
    warnings.warn(message, stacklevel=3)
  overrides["params"] = params
  return overrides
