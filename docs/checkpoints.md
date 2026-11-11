# Checkpoints

Policy checkpoints (`latest`, `best`, `final`) contain policy state and a
versioned architecture snapshot. They are suitable for inference or initializing
a new run. With `checkpoint_mode: weights` (the default), optimizer/replay and
training counters start fresh; saved normalization and target weights are kept.
Each requested seed receives an independent policy RNG.

Training writes `checkpoints/resume/` on its own cadence (`resume_every` env
steps, independent of `eval_every` -- the replay dump it carries, not the
eval, is what makes a save expensive) and again on normal completion. Each
save is an immutable, numbered generation (`resume/gen-000003/`, ...)
published via `resume/pointer.json` only once fully written; `resume_keep`
(default 2) is how many generations are then kept -- older ones are deleted
only after the new one is committed, so an interrupted or out-of-disk save
never leaves the previous generation as collateral damage. To continue an
interrupted run, use its original configuration with:

```yaml
checkpoint: /absolute/path/to/checkpoints/resume
checkpoint_mode: resume
```

Resume restores all seeds, optimizer moments/schedules, replay contents and RNGs,
normalization state, counters and best scores. The original training parameters,
seed count and replay capacity must match. Output paths, logging/evaluation
cadences, `logger:`, and `resume_every`/`resume_keep` may change. Saved model
snapshots take precedence over current defaults.

Environment episodes restart on resume. Partial action chunks are discarded;
completed replay transitions remain intact. This is training-state continuation,
not bit-identical simulator trajectory continuation. A failed iteration retains
the last completed resume generation; `final` policy weights may be newer.
Full replay checkpoints can be large, so account for their storage and I/O cost --
`save_training` refuses to start a new generation when free disk looks
insufficient to hold it alongside the previous one.

Use `with setup(config, num_seeds) as run:` when managing a run directly. The
public `train` and `validate` functions close owned resources on success or
failure, including failures during initialization and warmup.
