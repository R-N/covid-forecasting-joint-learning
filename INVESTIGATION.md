# Experiment Investigation

## Status

The prior experiment cannot support a conclusion that joint learning does or does not improve forecasting. Core implementation defects were corrected in `96a0104`, but the blockers below must be fixed before a rerun can compare methods. Improvement Opportunities collects the accuracy and cost work that is worth doing alongside that rerun but is not required for correctness.

## Fixes Applied

- The decoder now retains the most recent seed window, allowing predicted or teacher-forced rates and future exogenous features to reach later forecast steps.
- Continued validation reuse has been removed. The default is now to test the validation-selected checkpoint directly.
- Zero filling is causal, global features are scaled per group from training observations only, and local SIRD scaling and clustering use the training split only.
- OneCycleLR now advances for each successful optimizer update and resets after its configured total step count.
- Shared/private freezing uses the correct model instances and branch-specific modules; source loss is normalized so `source_weight` is the combined contribution of all sources.
- Cluster alignment now rejects sources without sufficient history before the target training split.
- CUDA conversion, Friedman statistics, ARIMA-SIRD metric scaling, SIRD forecast timing, and multi-step cumulative reconstruction have been corrected.
- Optuna now scores the checkpoint selected by early stopping; ARIMA has an explicit adapter for standard exogenous samples, and clustering-consistency runs use recorded distinct seeds.
- `requirements-experiment.txt` pins the Python 3.8/CUDA 11.1 experiment environment, including TensorBoard required by `SummaryWriter`.
- Group kabko removal now compares names correctly, `SourcePick.CLOSEST` supplies DTW with SIRD-rate arrays, and baseline Optuna searches use seeded samplers.
- Neural trials use distinct deterministic seeds and `main.optimize()` rejects parallel execution, which would race process-global RNG state.
- Model entrypoints default to CPU when `main.init()` was not called, and ARIMA treats `n_trials` as its total Optuna budget, matching SIRD.
- `main_1()` forwards `limit_data` to `main_0()`, so `limit_data=False` no longer leaves scalers and clustering silently split-limited while `preprocessing_4` honours the caller.
- `EarlyStopping.calculate_interval()` handles the empty first-epoch history by treating the current loss as the whole interval, covered by `tests/test_early_stopping.py`.

## Required Rerun Design

- Add a decoder regression test verifying that each generated output replaces the oldest seed entry.
- Add regression tests for preprocessing boundaries, scheduler steps, freezing, GPU evaluation, and baseline metrics. `tests/test_early_stopping.py` is the first of these and shows the pattern: plain `assert`, stubbed research dependencies, runnable as `python tests/<name>.py`.
- Use reconstructed IRD RMSSE consistently for early stopping, Optuna selection, and final reporting; tune and evaluate with identical epoch and scheduler budgets.
- Select hyperparameters and epochs on validation data once. If refitting on train plus validation, use the selected fixed epoch count without validation monitoring.
- Normalize source loss so all sources together have a configurable total weight, then test no-source and target-specific source-selection ablations.
- Compare private-only, hard-shared, joint, pooled, and naive baselines with equal search budgets over multiple seeds and rolling forecast origins.
- Verify corrected GPU evaluation, baselines, and statistical testing in the rerun before publishing comparisons.

## Remaining Blockers

- `pipeline/eval.py` computes Friedman chi-square incorrectly and uses signed one-sided post-hoc p-values.
- Optuna/early stopping select scaled SIRD-rate MSSE, while final neural results report reconstructed IRD RMSSE; optimization and final evaluation also have different default epoch schedules.
- Joint batches can overweight longer member datasets because batch truncation and loss normalization use different sample counts.
- ARIMA and SIRD baselines return scalar metrics where comparison logs require per-IRD values; the SIRD baseline also lacks a reliable adapter from standard pipeline data to three IRD-count columns.
- Unconstrained SIRD-rate outputs can reconstruct invalid compartment counts.
- Split boundaries assume zero forecast horizon while datasets use 14-day horizons, leaving the requested validation/test portions with far fewer valid forecast windows.
- ARIMA-SIRD cannot unpack the standard eight-field neural dataset correctly, and its exogenous path expects incompatible three-column inputs.

## Improvement Opportunities

Distinct from the blockers above: these are not correctness defects but accuracy and
cost improvements found while reading the model, training loop, and Optuna objective.
Several also remove failure modes, so they overlap with the rerun design.

### Quick wins — training and tuning cost

- `model/train.py:117` wraps the joint dataloader in `list(zip(...))`, materialising every batch of every member before each epoch. `zip` is already lazy; drop the `list()`.
- `pipeline/main.py:469` gives each member a `DataLoader` with `num_workers=0` over tensors that are already on the GPU, plus a `collate_fn` that re-stacks per batch. Splits are only a few hundred windows, so pre-stacking each split once and indexing with a random permutation removes the machinery entirely.
- `pipeline/preprocessing.py:212-216` and `:236-246` build every sliding window with per-window pandas `.iloc` and `to_numpy()`, and this is repeated for each Optuna trial. One `to_numpy()` plus `np.lib.stride_tricks.sliding_window_view` produces the same windows as views.
- `model/general.py:564` reruns `preprocessing_5`/`preprocessing_6` every trial and `:1305` deep-copies each group. Windowing depends only on past length, seed length, and the column sets; batching only on batch size. Memoise on that key.
- `model/general.py:1375` calls `posttrain_save_model()` inside the tuning objective, which computes captum attributions and writes four figures plus a spreadsheet per target per trial. Tuning never reads them; restrict it to the final evaluation path.
- `model/general.py:170` runs a learning-rate range test costing `0.5 * min_epoch` extra epochs per cluster per trial whenever `onecycle` is selected. Cache the result per architecture or search `lr` directly.
- `model/modules/main.py:358-381` recomputes the future representation over the whole seed window at every decoder step and uses only the last position, and reallocates the seed with `cat` plus a slice each step. Compute the new step incrementally over a preallocated buffer. Currently masked because `use_representation_future` defaults to false, which forces `seed_length` to 1.
- `model/general.py:185-192` builds OneCycleLR for `0.5 * min_epoch` epochs while `max_epoch` defaults to 150, so the schedule is exhausted and restarted mid-training. Derive its length from `max_epoch`.
- `model/general.py:206-216` hardcodes `clip_grad_norm_(..., 1)` with the autoclip implementation commented out, leaving `grad_clip_percentile` threaded through the constructor with no effect. Restore autoclip or drop the parameter.
- Automatic mixed precision is deliberately left off (`model/general.py:1019`, `:1315`), and the `autocast` path in `model/train.py:36` is ready for it. These models are small enough to be kernel-launch bound rather than FLOP bound, so AMP is not expected to pay for itself here.

### Quick wins — forecast accuracy

- Early stopping and the Optuna objective consume scaled SIRD-rate MSSE while results report reconstructed IRD RMSSE, so selection optimises a proxy of the reported metric through a nonlinear rebuild. Scoring the validation loader through the same `test()` path fixes it. This is the blocker above, listed here because it is also the cheapest accuracy gain.
- Bound the three SIRD-rate outputs, for example with `softplus`. This is required for valid compartment counts and additionally removes a class of `NaNPredException` trials.
- `source_weights` is searched over `(0.5, 1.0)` (`model/general.py:1171`), so the low-transfer regime is never explored. Widening to `(0.0, 1.0)` also yields a within-search no-transfer comparison.
- Teacher forcing is all-or-nothing per trial (`model/modules/main.py:393`, `:459-462`). Scheduled sampling, decaying the forcing probability across epochs, targets the exposure bias expected at a 14-step horizon.

### Big wins

- Batch the cluster members. `model/train.py:39-48` runs each member model sequentially, and each performs a 14-step autoregressive decode of small cells, so a training step is dominated by kernel-launch latency rather than compute. Member models are architecturally identical, so the member dimension can be folded into the batch with stacked per-member weights driven by `bmm` and grouped convolutions, with the shared branch run once over the concatenated batch. This is the largest available speedup. `torch.func.stack_module_state` with `vmap` would express it directly but needs a newer torch than the pinned 1.8.
- Enable Optuna pruning. The objective trains every cluster of every group to completion and never calls `trial.report()` or `should_prune()`, against a default `n_trials` of 10000. Reporting running validation loss with `HyperbandPruner`, or `MedianPruner` wrapped in `PatientPruner` given the noisy curves, cuts the budget several-fold. `NaNLossException` already subclasses `TrialPruned`.
- Shrink the search space. `past_strides`, `past_dilations`, `future_strides`, and `future_dilations` default to `(1, 1)` (`model/general.py:1160-1164`) but `TrialWrapper.suggest_int` treats any tuple as a range, so TPE models four constant dimensions; skip suggestion when the bounds are equal. Convolution depths of up to 20 over a 30-day window with kernels up to 14 overrun the window and are rejected by `check_conv_kwargs`, wasting trials. `TPESampler(multivariate=True, group=True)` suits this conditional space.
- Replace the recursive decoder with a direct multi-horizon head emitting all 14 steps at once. This removes exposure bias and the sequential launch cost together, and direct strategies are usually competitive at this horizon.
- Compute the loss on reconstructed IRD counts rather than scaled rates, which requires a differentiable torch `rebuild`. A rate error is harmless at low infected counts and severe at high ones, so this aligns the optimisation target with the evaluation target.
- Rotate the cluster target instead of fixing it to the shortest training series (`pipeline/main.py:303-310`). Leave-one-city-out within each cluster multiplies evaluation data at unchanged per-fit cost and tests the transfer claim in both directions.
- Use the existing branch-freezing hooks. `freeze_shared()` and `freeze_private()` (`model/general.py:223-229`) are never called from either training loop, so source and target gradients compete over the shared weights every step. Fitting the shared branch on sources, freezing it, then fitting private branches per target is the standard multi-task schedule and is cheap to trial.
- Evaluate over rolling forecast origins and multiple seeds. This gates every item above: on a single split and seed, run-to-run variation exceeds the effects being measured.

### Notes

- The default `n_trials` of 10000 in `main.optimize()` is not a reachable budget at current per-trial cost, even after the changes above.
- `model/general.py:1391` divides by `len(clusters)` while `:1310` skips clusters under `debug`, so debug-mode objective values are wrong.
