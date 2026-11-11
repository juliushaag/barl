"""Versioned policy checkpoints and reconstruction from saved model specifications.

`save_policy`/`load_policy` are the only two operations that actually construct
or restore learning state (an nnx `Policy`, an orbax array tree); `saved_model`
reads a checkpoint's `meta.json` and rebuilds its `ModelConfig` from nothing but
that JSON, so callers that only want metadata (`checkpoint_info.py`, config
diffing) can import this module and call it without pulling in jax/flax/orbax
at all -- those imports live inside the two functions that need them."""
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from barl_jax.compat import POLICY_VERSION, check_version, policy_class_for, restore_state


@contextmanager
def checkpoint_directory(path):
  path = Path(path).absolute()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
  backup = temporary.with_name(temporary.name + "-previous")
  committed = False
  try:
    yield temporary
    if path.exists():
      os.replace(path, backup)
    try:
      os.replace(temporary, path)
      committed = True
    except BaseException:
      if backup.exists():
        os.replace(backup, path)
      raise
  finally:
    shutil.rmtree(temporary, ignore_errors=True)
    if committed:
      shutil.rmtree(backup, ignore_errors=True)


def save_policy(policy, path):
  import orbax.checkpoint as ocp
  from flax import nnx

  meta = {
    "format_version": POLICY_VERSION,
    "model": policy.model_name,
    "policy_class": type(policy).__name__,
    "policy_type": f"{type(policy).__module__}:{type(policy).__qualname__}",
    "spec": {"schema_version": 1, "name": policy.model_name, "model": policy.config.to_dict()},
  }
  with checkpoint_directory(path) as destination:
    with ocp.StandardCheckpointer() as ckpt:
      ckpt.save(destination / "state", nnx.state(policy))
    (destination / "meta.json").write_text(json.dumps(meta, indent=2))


def saved_model(path, name=None):
  """`(ModelConfig, meta)` of a policy checkpoint, rebuilt from its `meta.json`
  alone -- the saved model is authoritative, current defaults never fill in."""
  from barl_jax.config.schemas import model_schema
  meta = json.loads((Path(path) / "meta.json").read_text())
  check_version(meta.get("format_version"), kind="policy")
  if name is not None and name != meta["model"]:
    raise ValueError(f"Checkpoint contains {meta['model']!r}, requested {name!r}")
  saved = meta.get("spec")
  if not isinstance(saved, dict) or "model" not in saved:
    raise ValueError(f"Checkpoint {path} has no versioned model spec")
  model = dict(saved["model"])
  # Checkpoints from before `chunk_size` moved into the params carry it as a kwarg.
  model["params"] = {**model.get("params", {}), **(meta.get("kwargs") or {})}
  return model_schema(meta["model"]).from_dict(model), meta


def load_policy(path, env, env_stats=None, name=None, **kwargs):
  from flax import nnx

  from barl_jax.policy import Policy
  path = Path(path).absolute()
  model, meta = saved_model(path, name)
  policy = Policy.make(meta["model"], env, model, policy_class=policy_class_for(meta),
                       env_stats=env_stats, **kwargs)
  nnx.update(policy, restore_state(path / "state", nnx.state(policy), meta["format_version"]))
  return policy

