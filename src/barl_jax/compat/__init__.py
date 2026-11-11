"""Backward compatibility -- the only place it lives.

Everything that lets older inputs keep working is in this package, so what is
still supported, and why, can be read in one sitting and deleted in one place:

- `config.py`  rewrites component config mappings written for an older schema
  (experiment YAML, `--set` overrides, the model spec inside a saved
  checkpoint) into the current one, before strict construction sees them.
  Each rule names the release it bridges.
- `checkpoint.py`  holds the checkpoint format versions and the stepwise state
  migrations (renamed parameter paths) that let `load_policy` and resume read
  a checkpoint written by an older version.

Nothing outside this package special-cases an old format: `config/schemas.py`
calls `upgrade_component`; `checkpoints.py` and `training_checkpoint.py` call
`check_version`, `restore_state` and `policy_class_for`. That is the whole
surface.
"""

from barl_jax.compat.checkpoint import (
  POLICY_VERSION,
  TRAINING_VERSION,
  check_version,
  policy_class_for,
  restore_state,
)
from barl_jax.compat.config import upgrade_component

__all__ = ["POLICY_VERSION", "TRAINING_VERSION", "check_version", "policy_class_for",
           "restore_state", "upgrade_component"]
