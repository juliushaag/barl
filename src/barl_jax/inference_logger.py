"""wandb logging for `scripts/evaluate.py` (inference over saved checkpoints).

Self-contained on purpose: this file plus the small hook in scripts/evaluate.py
and the optional `score=` argument of `evaluation.run_episodes` is the whole
feature, so it can be copied into another checkout. wandb is imported only
inside `InferenceLogger.__init__`.

Training logs one wandb run per *seed*; here the unit is the individual
*rollout*. Every evaluated episode becomes one run:

  group  = `<sweep>/<mode>`   -> wandb's group view averages over every seed's rollouts
  name   = `seedN/epK`        -> config carries seed, mode, episode, env, checkpoint

and carries a per-step curve on the x-axis `t`:

  reward, cum_reward                 what the env paid
  q                                  critic *expectation* Q(s_t, a_t) of the executed action
                                     (ensemble mean, distribution mean for categorical critics --
                                     not the pessimistic min used for acting/targets)
  entropy, alpha_entropy             policy entropy estimate H(s_t) and alpha*H (MaxEnt policies)
  return_to_go                       realised soft return: sum_k gamma^k r_{t+k}
                                     + sum_{k>=1} gamma^k alpha*H(s_{t+k}) -- the same
                                     entropy-in-target convention the critic is trained on
  return_to_go_reward                the reward-only part of it
  bias, bias_reward                  q - return_to_go, q - return_to_go_reward

plus summary fields `episode_return` (summed reward), `success`, `length`,
`terminated`, `q0`, `g0`, `bias0` (= q0 - g0, the cleanest single number per
rollout: Q at the start state vs. the realised discounted return) and
`mean_bias`.

Caveats worth keeping in mind when reading `bias`: the return-to-go is
truncated at the episode end, so on a time-limit cut (`terminated=False`) it
underestimates what the critic is predicting and `bias` reads high late in the
episode -- compare `bias0` or filter on `terminated`. `entropy` is the
log-prob of one fresh sample (an unbiased but noisy estimate of H), and for
diffusion actors costs a second sampling pass per step. Policies without a
critic log rewards only; policies without an entropy estimate (TD3, SDAC...)
get `alpha_entropy = 0`.
"""
import math
from pathlib import Path

import numpy as np


def make_probe_fn(policy):
  """A jitted `(observation, action, key) -> {"q": (B,), "entropy": (B,)?}`
  pinned to the policy's current weights in eval mode, or None if the policy
  has no critic. `q` is the critic's expectation at the clean noise level, on
  the normalized observation -- independent of `Policy.score`/`supports_bon`,
  so it exists for every critic-backed policy. `entropy` is present only when
  the actor reports one."""
  if getattr(policy, "critic", None) is None:
    return None
  from flax import nnx
  clone = nnx.clone(policy)
  clone.eval()

  # nnx.jit, not jax.jit over a closed-over state: the ensemble critic vmaps
  # its own members, which rejects nnx state captured from the outer trace.
  @nnx.jit
  def jitted(module, observation, action, key):
    obs = module.normalize_observations(observation)
    out = {"q": module.critic.expectation(obs, action)}
    _, entropy = module.actor.generate_action(obs, critic=module.critic, key=key)
    if entropy is not None:
      out["entropy"] = entropy
    return out
  return lambda observation, action, key: jitted(clone, observation, action, key)


def entropy_alpha(policy) -> float:
  """The coefficient on H in the critic's Bellman target: alpha for MaxEnt
  policies that put the entropy in the target, else 0."""
  if not hasattr(policy, "entropy") or not getattr(policy.params, "entropy_in_target", False):
    return 0.0
  return float(policy.alpha())


def critic_gamma(policy, default: float = 0.99) -> float:
  config = getattr(getattr(policy, "critic", None), "config", None)
  return float(getattr(config, "gamma", default))


def return_to_go(rewards, gamma: float, bonus=None) -> np.ndarray:
  """Discounted suffix sums of `rewards`. `bonus[t]` (alpha*H at the state
  *after* step t) is discounted one step further, as in the soft Bellman target
  r + gamma * (Q' + alpha*H')."""
  out = np.zeros(len(rewards))
  running = 0.0
  for t in range(len(rewards) - 1, -1, -1):
    step = rewards[t] + (gamma * bonus[t] if bonus is not None else 0.0)
    running = step + gamma * running
    out[t] = running
  return out


def _finite(value):
  return float(value) if value is not None and math.isfinite(value) else None


class InferenceLogger:
  def __init__(self, project: str = "BA-RL", group: str = None, tags: list[str] = None,
               entity: str = None, finish_timeout: float = 120.0):
    import wandb
    self._wandb = wandb
    self._project = project
    self._group = group
    self._tags = list(tags or [])
    self._entity = entity
    self._finish_timeout = finish_timeout

  def log_checkpoint(self, checkpoint: Path, *, sweep_root: Path, env_name: str, entry: dict,
                     records: dict, gamma: float = 0.99, alpha: float = 0.0) -> None:
    """`records` is `{mode: [episode record, ...]}` straight from
    `run_episodes` (rewards, probe, success, terminated). Never raises: a wandb
    outage must not cost an evaluation that already ran -- the JSON output is
    the source of truth. Failures are per rollout, so one bad run does not
    drop its siblings."""
    seed = next((p.name for p in Path(checkpoint).parents if p.name.startswith("seed")), "seed?")
    failures = 0
    for mode, episodes in records.items():
      for k, record in enumerate(episodes):
        try:
          self._log_rollout(checkpoint, sweep_root, env_name, entry, mode, seed, k, record, gamma,
                           alpha)
        except Exception as exc:
          failures += 1
          if failures == 1:
            print(f"  [wandb] logging {checkpoint} [{mode}] ep{k} failed -- "
                  f"{type(exc).__name__}: {exc}")
    if failures:
      print(f"  [wandb] {failures} rollout(s) of {checkpoint} not logged")

  def _log_rollout(self, checkpoint, sweep_root, env_name, entry, mode, seed, k, record, gamma,
                   alpha):
    wandb = self._wandb
    rewards = np.asarray(record["rewards"], np.float64)
    probe = record.get("probe") or {}
    q = np.asarray(probe.get("q", []), np.float64)
    has_q = len(q) == len(rewards) and len(q) > 0
    entropy = np.asarray(probe.get("entropy", []), np.float64)
    has_h = len(entropy) == len(rewards) and len(entropy) > 0 and alpha != 0.0
    cum = np.cumsum(rewards)
    # alpha*H at the state reached by step t = the entropy probed at step t+1
    # (0 past the last step: nothing bootstraps from a terminal state).
    bonus = np.append(alpha * entropy[1:], 0.0) if has_h else None
    g_reward = return_to_go(rewards, gamma)
    g = return_to_go(rewards, gamma, bonus) if has_h else g_reward
    bias = q - g if has_q else None

    config = {key: v for key, v in entry.items() if key not in ("env", "mode")}
    digits = seed.removeprefix("seed")
    config.update(env=env_name, mode=mode, seed=int(digits) if digits.isdigit() else seed,
                  episode=k, gamma=gamma, alpha=alpha, checkpoint_path=str(checkpoint), sweep=sweep_root.name)
    run = wandb.init(project=self._project, entity=self._entity,
                     group=f"{self._group or sweep_root.name}/{mode}", name=f"{seed}/ep{k}",
                     job_type="inference", tags=["inference", env_name, mode, seed, *self._tags],
                     config=config, reinit="create_new",
                     settings=wandb.Settings(finish_timeout=self._finish_timeout))
    try:
      run.define_metric("t")
      run.define_metric("*", step_metric="t")
      for t in range(len(rewards)):
        row = {"t": t, "reward": float(rewards[t]), "cum_reward": float(cum[t]),
               "return_to_go": float(g[t]), "return_to_go_reward": float(g_reward[t])}
        if len(entropy) == len(rewards):
          row.update(entropy=_finite(entropy[t]), alpha_entropy=_finite(alpha * entropy[t]))
        if has_q:
          row.update(q=_finite(q[t]), bias=_finite(bias[t]),
                     bias_reward=_finite(q[t] - g_reward[t]))
        run.log(row)
      summary = {"episode_return": float(cum[-1]) if len(cum) else 0.0, "length": len(rewards),
                 "success": bool(record["success"]), "terminated": bool(record["terminated"]),
                 "act_nonfinite": int(record.get("act_nonfinite", 0))}
      if has_q:
        summary.update(q0=_finite(q[0]), g0=float(g[0]), bias0=_finite(q[0] - g[0]),
                       bias0_reward=_finite(q[0] - g_reward[0]),
                       mean_bias=_finite(np.nanmean(bias)))
      for key, value in summary.items():
        run.summary[key] = value
    finally:
      run.finish()
