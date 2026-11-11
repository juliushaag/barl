# Per-rollout inference logging to wandb — implementation spec

Audience: a Claude instance applying this to another checkout of this repo (`barl_jax`, `scripts/evaluate.py`, `scripts/submit.py`). Line numbers refer to the source repo and may drift; match by the quoted code.

## Goal
`scripts/evaluate.py` evaluates saved checkpoints (a sweep dir → every `seed*/best` checkpoint) under several action executors (modes). Make it log to wandb **by default**, with **one wandb run per rollout (episode)**, not per seed/checkpoint. Each run carries per-step reward, summed reward, critic Q expectation, policy entropy and an overestimation estimate. The cluster path (`submit.py --eval-config` → `evaluate.py --config <file>`) must log with no extra flag. Episodes default drops from 10/20 to 5.

## Design decisions (do not change)
- Run name `seedN/epK`; group `<sweep dir name>/<mode>`; `job_type="inference"`; tags `inference`, env, mode, seedN. Config holds seed (int), mode, episode, env, gamma, alpha, checkpoint_path, sweep, plus the eval entry's other keys.
- Per-step x-axis `t` (`run.define_metric("t")`, `define_metric("*", step_metric="t")`).
- `q` is the critic **expectation** (`critic.expectation`: ensemble mean; distribution mean for categorical critics) at the clean noise level on the normalized observation — NOT `Policy.score`/`value` (that applies the pessimistic `value_reduce`).
- Soft return-to-go matches the critic target `r + gamma*(Q' + alpha*H(s'))` (`critics/base.py: scalar_td_target`): `G_t = sum_k gamma^k r_{t+k} + sum_{k>=1} gamma^k * alpha*H(s_{t+k})`. alpha*H of the state reached after step t is the entropy probed at step t+1; 0 past the last step. alpha is nonzero only for policies with an `entropy` attribute and `params.entropy_in_target`.
- Probing must not change the action RNG stream: the probe key is `jax.random.fold_in(subkey, 1)` where `subkey` is the key given to `act_on`.
- Logging failures never lose an evaluation: `InferenceLogger.log_checkpoint` catches per rollout and prints one warning.
- JSON output stays aggregates-only (mean/std return, success rate). Per-step data goes only to wandb.
- Do NOT change `Critic.supports_bon` (NCLQL and noise-conditioned critics have no Best-of-N scorer; `tests/test_critics.py:132` and `tests/test_executors.py:613` assert that). Q for every critic-backed policy comes from the probe, independent of `supports_bon`. (Optional follow-up the user may request: make `supports_bon` return True and update those two tests — no documented reason for the gate was found.)

## Changes

### 1. New file `src/barl_jax/inference_logger.py`
Create it with exactly this content:

```python
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
```

### 2. `src/barl_jax/evaluation.py` — `run_episodes`
- Add `import jax.numpy as jnp`.
- Signature: add `probe=None` keyword (before `**inference_kwargs`). Document it: `probe(observation, action, key) -> {name: (batch,) or (batch,1)}` diagnostics at every stepped (state, executed action); each record gets `probe: {name: [value per step]}` aligned with `rewards`; not supported for StackedPolicy (rows is None).
- Per-slot record dict: add `probe={}` (e.g. `dict(frames=[], rewards=[], probe={}, terminated=False, ...)`).
- Right after `nonfinite = ~np.isfinite(action_np)` and before `step_env`:
```python
probed = None
if probe is not None and rows is not None:
  probed = {name: np.asarray(v).reshape(len(rows), -1)[:, 0] for name, v in probe(
    jnp.asarray(obs[rows], jnp.float32), jnp.asarray(action_live),
    jax.random.fold_in(subkey, 1)).items()}
```
- In the per-live-slot loop, next to `data[i]["rewards"].append(reward[i])`:
```python
if probed is not None:
  for name, values in probed.items():
    data[i]["probe"].setdefault(name, []).append(float(values[np.searchsorted(rows, i)]))
```
Note `obs[rows]`/`action_live` are the observation the action was chosen at and the raw (pre-`scale_action`) policy action — the same pair the critic was trained on.

### 3. `scripts/evaluate.py`
- `evaluate(..., detail=None)`: after creating `vector`, if `detail is not None`:
```python
from barl_jax.inference_logger import critic_gamma, entropy_alpha, make_probe_fn
extra["probe"] = make_probe_fn(policy)
detail["gamma"] = critic_gamma(policy)
detail["alpha"] = entropy_alpha(policy)
detail["records"] = {}
```
  (`extra = {}` is defined before.) Pass `**extra` to `run_episodes(...)` and, after it, `if detail is not None: detail["records"][mode] = records`. Pass `probe` only when logging — tests monkeypatch `run_episodes` with signature `(actor, vector, episodes, *, render, seed)`.
- `evaluate_entry(entry, logger=None)`: `detail = {} if logger is not None else None`; pass `detail=detail` to `evaluate(...)`; after it:
```python
if logger is not None and detail.get("records"):
  logger.log_checkpoint(path, sweep_root=root, env_name=env_name, entry=entry,
                        records=detail["records"], gamma=detail["gamma"],
                        alpha=detail["alpha"])
```
  (`root` is the already-resolved `Path(entry["path"]).resolve()`.)
- `DEFAULTS["episodes"]`: 10 → 5.
- argparse: `--wandb` as `argparse.BooleanOptionalAction, default=True` (gives `--no-wandb`), `--wandb-project` (default `"BA-RL"`), `--wandb-group` (default None → sweep dir name).
- `main`: before the "Evaluating N entries" print, `logger = None; if args.wandb: from barl_jax.inference_logger import InferenceLogger; logger = InferenceLogger(args.wandb_project, group=args.wandb_group)`. Call `evaluate_entry(entry, logger) if logger is not None else evaluate_entry(entry)`.
- The `--config` mutual-exclusion check only looks at `DEFAULTS` keys, so `--wandb*` flags are allowed alongside `--config`; leave it.

### 4. Episode defaults
- `scripts/submit.py`: `--episodes` default 10 → 5 (the `ev.add_argument("--episodes", ...)` line) and the matching `args.episodes != 10` check → `!= 5`.
- `experiments/eval.yaml`: `episodes: 20` → `5`, and the header comment's `episodes: 10` → `5`.

### 5. Tests (`tests/test_evaluate.py`)
- Autouse fixture `monkeypatch.setenv("WANDB_MODE", "disabled")` (main() now logs by default).
- `test_main_writes_nan_results_instead_of_losing_the_file`: change the `evaluate_entry` monkeypatch lambda to `lambda entry, logger=None: ...`.
- Add: `test_inference_logger_logs_one_run_per_rollout_with_critic_bias` — save a SAC policy on the file's `ShortPendulum` family under `tmp/sweep/seed3/best`, call `module.evaluate(..., ["DirectExecutor","MeanExecutor"], episodes=2, num_envs=2, num_candidates=2, seed=0, detail=detail)`; assert each record has `probe["q"]` and `probe["entropy"]` of `len(rewards)`; then monkeypatch `sys.modules["wandb"]` with a fake (`init(**kw)` returning a recording run with `define_metric/log/summary/finish`, and `Settings(**kw)`), call `InferenceLogger(project="p").log_checkpoint(...)`, and assert 4 runs, `group == "sweep/DirectExecutor"`, `name == "seed3/ep0"`, `config["seed"] == 3`, first row has `t, reward, cum_reward, q, entropy, return_to_go, bias`, `summary["episode_return"] ≈ last cum_reward`, `summary["bias0"] ≈ q[0] - return_to_go[0]`.
- Add: `return_to_go([1,1,1], 0.5) == [1.75, 1.5, 1.0]` and `return_to_go([1,1], 0.5, [2,0]) == [2.5, 1.0]`.

## Verification
```bash
uv run ruff check src scripts tests/test_evaluate.py
BARL_ALLOW_CPU=1 uv run pytest tests/test_evaluate.py tests/test_eval_actions.py tests/test_eval_rounds.py tests/test_eval_seeding.py -q
```
Sanity check that the probe works for every policy (expect `q` for all; `entropy` only for SAC/DIME/AMDP; alpha > 0 only for those three):
```python
import gymnasium as gym, jax, jax.numpy as jnp
from barl_jax.policy import Policy
from barl_jax.inference_logger import make_probe_fn, entropy_alpha
env = gym.make("Pendulum-v1")
for name in Policy.available():
    p = Policy.make(name, env, seed=0)
    print(name, {k: v.shape for k, v in make_probe_fn(p)(jnp.zeros((4,3)), jnp.zeros((4,1)), jax.random.PRNGKey(0)).items()}, entropy_alpha(p))
```
A real end-to-end check needs a trained sweep and wandb credentials: `uv run python scripts/evaluate.py sweeps/<sweep> --mode DirectExecutor MeanExecutor --episodes 2`, then confirm runs appear in group `<sweep>/<mode>`.

## Known caveats (document, don't fix)
- Run count = seeds × modes × episodes (each ~1–2 s to init); 10 seeds × 7 modes × 5 episodes = 350 runs.
- `return_to_go` is truncated at episode end, so on a time-limit cut (`terminated=False`) `bias` reads high late in the episode; use `bias0` or filter on `terminated`.
- `entropy` is −log p of one fresh sample (noisy); for DIME/AMDP it costs an extra diffusion sampling pass per step. SDAC, SDACPaper, TD3, NCLQL report no entropy (alpha*H = 0).
- Cluster jobs need wandb credentials (`WANDB_API_KEY` or prior login). `submit.py` forwards no wandb flags, so project/group are the defaults there (`BA-RL`, sweep dir name).
