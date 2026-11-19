"""Checkpoint format versions and the state migrations between them.

A migration upgrades a checkpoint written at `version` to `version + 1` by
renaming state paths, inside one component slot or from the policy root. Loading never rewrites the
files: the current state tree is mapped back to the saved layout (renames
inverted, newest first), orbax restores into that template, and the values land
in the current tree. One rename list therefore serves policy checkpoints and
resume checkpoints alike -- an optimizer's moments sit under the same
`critic/...` paths as the parameters they track.

To change a saved layout: bump `POLICY_VERSION` / `TRAINING_VERSION`, append a
`StateMigration` for the old version, and add a fixture under
`tests/fixtures/policy_v<old>/` written by the old code.
"""

from dataclasses import dataclass
from pathlib import Path

POLICY_VERSION = 5
TRAINING_VERSION = 5

Renames = tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]


@dataclass(frozen=True)
class StateMigration:
  version: int        # reads a checkpoint written at this version ...
  description: str    # ... and states what changed on the way to version + 1
  slot: str | None    # the component slot whose subtree is renamed (None: the policy root)
  # (old subpath, new subpath) under `slot`. Several new paths may name the same
  # old one: each is restored from it.
  renames: Renames


MIGRATIONS = (
  StateMigration(
    version=3,
    description="The critic became a composition of parts: its target network, "
                "value head (with its support) and training objective moved under "
                "`target`, `head` and `objective`.",
    slot="critic",
    renames=(
      (("net_target",), ("target", "net_target")),
      (("target_steps",), ("target", "steps")),
      (("support_net", "endpoints"), ("head", "support", "endpoints")),
      (("dual",), ("head", "support", "dual")),
      (("support",), ("head", "support", "centers")),
      (("edges",), ("head", "support", "edges")),
      (("noise", "sigmas"), ("objective", "sigmas")),
    ),
  ),
  StateMigration(
    version=4,
    description="Observation normalization moved from the Policy into an `input_norm` part "
                "of every actor and critic; each starts from the statistics the Policy kept "
                "(see `compat.config.upgrade_model` for the matching config).",
    slot=None,
    renames=tuple(
      ((f"obs_{stat}",), (slot, "input_norm", stat))
      for slot in ("actor", "critic") for stat in ("mean", "std", "count")
    ),
  ),
)

OLDEST_POLICY_VERSION = min([m.version for m in MIGRATIONS] + [POLICY_VERSION])
OLDEST_TRAINING_VERSION = min([m.version for m in MIGRATIONS] + [TRAINING_VERSION])


def check_version(version, *, kind: str = "policy") -> int:
  """`version` if a checkpoint of that format can be read, else a clear error."""
  current, oldest = ((POLICY_VERSION, OLDEST_POLICY_VERSION) if kind == "policy"
                     else (TRAINING_VERSION, OLDEST_TRAINING_VERSION))
  if not isinstance(version, int) or not oldest <= version <= current:
    raise ValueError(f"Unsupported {kind} checkpoint version {version!r}: this release reads "
                     f"versions {oldest}..{current} (see barl_jax/compat/checkpoint.py)")
  return version


def _rename(path: tuple, slot: str | None, renames, *, inverse: bool) -> tuple:
  """`path` with its first matching subpath renamed (under `slot`, when given).
  Longest match first, so a nested rename wins over its parent's. Matches may
  start anywhere: a resume checkpoint nests the policy under its own prefix."""
  prefix = (slot,) if slot is not None else ()
  pairs = [(prefix + new, prefix + old) if inverse else (prefix + old, prefix + new)
           for old, new in renames]
  pairs.sort(key=lambda pair: -len(pair[0]))
  for i in range(len(path)):
    for src, dst in pairs:
      if path[i:i + len(src)] == src:
        return path[:i] + dst + path[i + len(src):]
  return path


def saved_path(path: tuple, from_version: int) -> tuple:
  """Where a current state path was stored by a checkpoint of `from_version`."""
  path = tuple(str(p) for p in path)
  for migration in sorted(MIGRATIONS, key=lambda m: -m.version):
    if migration.version >= from_version:
      path = _rename(path, migration.slot, migration.renames, inverse=True)
  return path


def restore_state(directory, state, from_version: int):
  """Restore the orbax tree at `directory` (written at `from_version`) into
  `state`, an nnx State in the current layout; returns that state filled in."""
  import orbax.checkpoint as ocp
  from flax import nnx

  flat = list(nnx.to_flat_state(state))
  template = {}
  for path, variable in flat:
    template[saved_path(path, from_version) + ("value",)] = variable[...] \
      if not _is_key(variable) else variable.value
  item = nnx.traversals.unflatten_mapping(template)
  with ocp.PyTreeCheckpointer() as ckpt:
    restored = ckpt.restore(
      Path(directory),
      args=ocp.args.PyTreeRestore(
        item=item, restore_args=ocp.checkpoint_utils.construct_restore_args(item),
        partial_restore=True))
  values = nnx.traversals.flatten_mapping(restored)
  for path, variable in flat:
    variable.value = values[saved_path(path, from_version) + ("value",)]
  return nnx.from_flat_state(flat)


def policy_class_for(meta) -> str:
  """The policy class to rebuild a checkpoint with: its saved `module:Class`,
  unless the class has since moved -- then its registered name, which every
  built-in keeps across moves."""
  from barl_jax.registration import import_type
  try:
    import_type(meta["policy_type"])
    return meta["policy_type"]
  except (ImportError, AttributeError, KeyError, TypeError):
    return meta["policy_class"]


def _is_key(variable) -> bool:
  import jax
  value = variable.value
  return hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
