# Value Forcing critic

`ValueForcingCritic` implements the critic from *Value Forcing: Aligning Return
Distributions at All Noise Levels* (Zheng et al.,
[arXiv:2611.02160](https://arxiv.org/abs/2611.02160)).

```bash
uv run python scripts/submit.py --algos SAC --envs HalfCheetah --seeds 3 \
  --set model.components.critic.class=ValueForcingCritic
```

It works with every actor that prices actions through `critic.value()` (SAC,
TD3, AMDP, DIME, SDAC, SDACPaper) and with the best-of-N executors. NCLQL needs a
noise-conditioned critic and is not supported.

## What it learns

The return distribution is a flow. A velocity field `v(z | s, a, tau)`
transports `z0 ~ N(0, 1)` at noise level `tau = 0` to a return sample at
`tau = 1`. Earlier flow critics supervise only the endpoint; Value Forcing
applies a Bellman backup at every noise level, the *flow Bellman equation*
(Eq. 5 of the paper):

```
Z(s, a, tau)  =d  tau * r + gamma * Z(s', a', tau) + (1 - tau) * sqrt(1 - gamma^2) * N
```

The Gaussian term `N` keeps both sides standard normal at `tau = 0`; at
`tau = 1` the equation is the ordinary distributional Bellman equation.
Differentiating in `tau` gives the velocity's regression target (Eq. 13).

## What is implemented

The paper's Q-velocity-only variant ("VF without the V-velocity", App. D.5),
which its ablation reports as performing similarly to the full method: the
Q-flow bootstraps from its own target copy at the policy's next action. The
paper's V-velocity, flow behavior-cloning policy, rejection sampling and the
implicit variant (IVF) are offline-RL machinery and are not part of this critic.

The backup is generalised to this repo's transitions. With
`g = (1 - terminated) * gamma^horizon` and `R` the chunk return:

```
z_in   = tau * (R + g * bonus) + g * z_tau + (1 - tau) * sqrt(1 - g^2) * eps
target =        R + g * bonus  + g * v_target(z_tau | s', a', tau) - sqrt(1 - g^2) * eps
loss   = (v(z_in | s, a, tau) - target)^2
```

`z_tau` is the target flow integrated from `z0` to `tau` at `(s', a')`, and
`bonus` is the entropy bonus of max-entropy policies. A terminal transition
(`g = 0`) flows straight from noise to `R`.

## Settings

| Field | Default | Meaning |
|-------|---------|---------|
| `flow_steps` | 10 | Euler steps bringing the bootstrap sample from `z0` to `z_tau` (paper: 10) |
| `value_steps` | 1 | Euler steps of the Q estimate; 1 is the paper's training-time one-step estimate |
| `value_samples` | 8 | return samples averaged into the Q estimate |
| `return_scale` | `null` → `1 - gamma` | returns are modelled as `return_scale * reward_scale * G` (the paper's reward scaling) |
| `time` | 16-dim sinusoidal | noise-level embedding (paper: 16-dim sinusoidal) |
| `target` | `Polyak`, `tau: 0.005` | the bootstrap flow (paper: 0.005) |
| `ensemble` | 2 members, `mean` / `mean` | see below |
| `mlp` | GELU MLP with LayerNorm, 512 × 3 | the paper uses 1024 × 4 |

`value()` converts back from normalised units, so actors, entropy terms and
logged Q-values are ordinary reward-scaled returns.

## Differences from the paper

- **Deterministic Q estimate.** `value()` takes no random key, so the estimate
  integrates from fixed Gaussian quantile points instead of random `z0` draws.
- **Ensemble.** The paper uses two members "as in the twin Q trick" without
  spelling out the bootstrap. Here `target_reduce: mean` (default, as for every
  critic in this repo) lets each member bootstrap from its own target copy, and
  `target_reduce: min` makes every member bootstrap from the target member with
  the lower value estimate at `(s', a')`.
- **Online use.** The paper evaluates offline RL only. Using the critic online
  with these actors is this repo's adaptation and has not been benchmarked here.
- **No batch-statistic norms.** `(s, a)` and `(s', a')` are evaluated at
  different noisy returns, so they cannot share batch statistics; `batch` and
  `batchrenorm` are rejected.

## Sampling the distribution

`critic.return_samples(s, a, key, samples, steps)` draws from the learned return
distribution. Euler integration reproduces the mean at any step count but
underestimates the spread when steps are few: with an exact velocity field for a
Gaussian return, the sampled std is 0 at 1 step, about 12% low at 10 steps and
within 1% at 512. The one-step Q estimate used by `value()` is therefore
unaffected, but use many steps when the shape of the distribution matters.

## Tests

`tests/test_value_forcing.py` verifies the backup against a closed-form solution:
for a self-loop with Gaussian rewards the noisy return at level `tau` is
`N(tau * M, tau^2 S^2 + (1 - tau)^2)`, which fixes the exact velocity field, and
that field must be a fixed point of the implemented backup (also with n-step
returns, an entropy bonus, return scaling and terminal transitions). It also
recomputes the loss by hand from the same random draws, and checks that the critic learns action-dependent values, converges to
`1 / (1 - gamma)` on a constant-reward self-loop (reachable only by
bootstrapping), and recovers a two-outcome return distribution instead of
collapsing to its mean. `critic.return_samples(s, a, key)` draws from the
learned distribution.
