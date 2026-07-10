# Experiment Investigation

## Status

The prior experiment cannot support a conclusion that joint learning does or does not improve forecasting. Core implementation defects were corrected in `96a0104`, but the blockers below must be fixed before a rerun can compare methods.

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

## Required Rerun Design

- Add a decoder regression test verifying that each generated output replaces the oldest seed entry.
- Add regression tests for preprocessing boundaries, scheduler steps, freezing, GPU evaluation, and baseline metrics.
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
- `main_1(limit_data=False)` does not forward `limit_data` to `main_0`, allowing later SIRD scaling to include validation/test observations.
- Unconstrained SIRD-rate outputs can reconstruct invalid compartment counts.
- `EarlyStopping.__call__()` computes intervals before recording the first losses, so the default interval helper can raise `IndexError` on the first epoch.
- Split boundaries assume zero forecast horizon while datasets use 14-day horizons, leaving the requested validation/test portions with far fewer valid forecast windows.
- ARIMA-SIRD cannot unpack the standard eight-field neural dataset correctly, and its exogenous path expects incompatible three-column inputs.
