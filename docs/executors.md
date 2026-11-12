# Action selection at inference time

`RunConfig.collection_executor`/`eval_executor` each select, independently,
an `ActionExecutor` (`barl_jax/executors/`) that sits between a policy's
candidate generation and the env, at every inference call -- policy-
independent (no executor branches on SAC/AMDP/DIME/SDAC/NCLQL). Each is a
registered component (`class:` in YAML, `get_executor()` at runtime) with
its own typed config declaring only the fields it uses:

- `DirectExecutor` (the default) -- one unchanged inference call, bit-
  identical to not passing an executor at all; its config has no
  `num_candidates` field at all, so it cannot be pointed at one.
- `MeanExecutor` (`num_candidates`) -- samples `num_candidates` action plans
  and averages them elementwise. Continuous (box) action spaces only; can
  land between modes of a multimodal policy, so treat it as a research
  strategy, not an automatic improvement.
- `MedianExecutor` (`num_candidates`) -- like `MeanExecutor`, but the
  elementwise median: unmoved by a single wild candidate the way an average
  would be. Continuous action spaces only.
- `AxisShuffleExecutor` (`num_candidates`) -- samples `num_candidates` plans
  and, independently for every `(batch, act_dim)` entry, takes the
  value from a randomly chosen candidate -- a per-axis recombination of all
  N, not any one candidate's action. Needs no critic; works for discrete
  action spaces too (it only ever copies a real candidate's value, never
  blends one).
- `BestOfNExecutor` (`num_candidates`) -- samples `num_candidates` plans and
  keeps the highest-scoring one under the policy's own critic (`Policy.score`,
  `Policy.supports_bon` -- SAC/AMDP/DIME/SDAC via `RLPolicy`; NCLQL has none
  and is rejected at preflight with an actionable message, same as any
  algorithm without a scorer).
- `TopKExecutor` (`num_candidates`, `top_k`) -- BoN generalized: averages the
  `top_k` highest-scoring candidates instead of hard-selecting one, softening
  BoN's failure mode of exploiting a miscalibrated critic's noise on a single
  winner. `top_k=1` behaves like `BestOfNExecutor`; `top_k=num_candidates`
  behaves like `MeanExecutor`. Needs a scorer; blends elementwise
  (continuous-only) whenever `top_k > 1`.
- `PerturbExecutor` (`num_candidates`, `sigma`) -- one base action plus
  `num_candidates - 1` local Gaussian perturbations of it (std `sigma`,
  clipped back to a valid action), scored and BoN-selected. Since the base
  action is always in the pool, the winner can never score worse than not
  perturbing at all -- a local search around the policy's own proposal,
  handy when independent resampling (`BestOfNExecutor`) barely moves the
  action (e.g. a near-deterministic policy late in training). Needs a scorer.

All are validated before any resource is allocated: an unknown `class` or a
bad `num_candidates`/`top_k`/`sigma` value in `barl_jax.config.run.validate_run`
(each executor config's own `__post_init__`/unknown-field rejection), and the
capability/action-space checks (`ActionExecutor.validate`) as soon as
`trainer.setup()` has a constructed reference policy. `collection_executor` is
part of the training config (changing it is a behavior-policy change, so
resume rejects a mismatch); `eval_executor` may change across a resume.
