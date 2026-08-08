# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code (undergrad thesis + published paper) for COVID-19 forecasting of East Java (Jatim) regencies/cities using **joint learning** (multi-task learning with shared + private branches). It is a `pip`-installable Python package, driven from notebooks/scripts that import it — there is no CLI, and `tests/` holds standalone regression checks rather than a test suite. Companion Streamlit web app lives in a separate repo (see README). Current experiment results are not valid evidence of model performance: split horizons, metric selection, baseline compatibility, and statistical testing remain unresolved. Optuna checkpoint scoring, sequential optimization seeds, generic ARIMA sample handling, clustering/source selection, and SIRD reconstruction bounds have been corrected; `INVESTIGATION.md`'s "Must do" and "Quick wins" items are now applied at the code level too (naive/Theta/linear/GBT comparison baselines, scheduled sampling, spectral-entropy forecastability, clustering spread, MinT reconciliation with shrinkage, fused `PastHead`, median-seed ensembling), and its 6 "Big wins" all have real progress: reconstructed-IRD training loss, cluster-target rotation, alternating branch-freeze schedule, and a direct multi-horizon decoder head are fully applied; `model/modules/fused.py` fuses the whole Linear-only direct-decoder stage across cluster members via HFTA and is verified end-to-end against real `SingleModel` instances (forward-pass correctness only -- not yet wired into `ClusterModel`'s live training loop, and no CUDA GPU here to measure the actual throughput payoff); `scripts/epicastbench_joint_learning_check.py` actually trains `SingleModel`'s real private+shared architecture on real downloaded EpiCastBench data and reports real forecast RMSSE against `scripts/epicastbench_check.py`'s naive/linear/theta/gbt baselines (the untuned joint model loses to every baseline there — an honest, expected-at-this-scale result, not a defect). None of the neural-model changes have been run against real GPU-scale/Optuna-tuned training yet, since no environment used to make these changes has torch with CUDA. See `INVESTIGATION.md` before reproducing or extending them; its `Recommendations` section is the consolidated priority list (must-do, quick wins, big wins, rejected), and the graded literature review below it records the evidence and the caveats behind each item.

Key terminology: **kabko** = *kabupaten/kota* = an Indonesian regency/city, the fundamental data unit.

## Commands

Windows-first (`.bat` helpers), but the underlying commands are cross-platform:

```bash
python -m pip install --user --upgrade setuptools wheel   # prebuild.bat
pip install -r requirements-experiment.txt                # Python 3.8 + CUDA 11.1 experiment environment
pip install -e .                                           # install.bat (editable install)
python setup.py sdist bdist_wheel                          # build.bat (build wheel)
```

There is **no linter and no run entry point**, and no test runner: `tests/` contains standalone `assert`-based scripts run individually (`python tests/test_early_stopping.py`), each stubbing the heavy research dependencies so it works in a bare interpreter. Follow that pattern when adding checks. You otherwise exercise the code by importing `covid_forecasting_joint_learning` and calling the pipeline stage functions (see below). `requirements-experiment.txt` pins the Python 3.8/CUDA 11.1 research environment, including `torch==1.8.1+cu111`; `orange3` (Orange) is required by `pipeline/eval.py`.

## Data flow

Input is an Excel workbook with named sheets (`covid_indo`, `covid_jatim`, `vaccine`, `test`, `population`, `psbb`, `ppkm`, `ppkm_mikro`, `long_holiday`, `pilkada`, `other_dates`) loaded by `data/center.py::DataCenter.load_excel`. Alternatively `data/getter.py::DataGetter` pulls raw daily data from an HTTP endpoint. `data/drive.py` (pydrive2) syncs logs/models to Google Drive during optimization.

The domain model is **SIRD** (Susceptible/Infected/Recovered/Dead compartments). Raw case counts are converted in `pipeline/sird.py` into the SIRD variables the network actually predicts: `SIRD_VARS = [beta, gamma, delta]` (`data/cols.py`). The network outputs these vars; `SingleModel.rebuild` + `pipeline/sird.py` reconstruct absolute IRD counts. `sird.rebuild` clamps the rates and the flows between compartments, since nothing bounds a predicted rate. `model/torch_sird.py` is a differentiable torch port of that same recurrence, used by the opt-in `model/loss.py::ReconstructedRMSSELoss` to train on reconstructed-count error instead of scaled-rate error (default loss is unchanged). All column-name constants live in `data/cols.py` — reuse them, don't hardcode strings.

## Pipeline (the spine of the project)

`pipeline/main.py` exposes **numbered stage functions** run in order; `main.py::main_0`/`main_1` orchestrate them. The numbering *is* the execution order:

- `preprocessing_0` — global-timeseries preparation
- `get_kabkos` / `preprocessing_1` — build per-kabko `KabkoData`, compute SIRD vars
- `preprocessing_2` — **split into groups** by series length (`limit_length`, `limit_date`) and compute train/val/test `split_indices`
- `preprocessing_3` — fit per-group training-only scalers; `main_0` also scales non-date global features from the training portion
- `clustering_1` — **DTW time-series K-Means** (tslearn) on each group's training portion → `Cluster`s. Single-member clusters are dropped as outliers; the **shortest training** series in each cluster becomes the `target`, the rest are `sources` (`Cluster.rotate_targets()` yields one alternative `Cluster` copy per member turned target instead, for evaluating every member instead of just the shortest one).
- `preprocessing_4` — per-cluster rescale aligned to target
- `preprocessing_5/6/7` — build sliding-window datasets, wrap in torch `DataLoader`s (5/6 = train, 7 = future prediction)

Post-hoc analysis utilities, independent of the training loop above: `pipeline/eval.py` (`ensemble_eval_logs` median-combines per-seed `EvalLog`s, `forecastability_by_kabko` computes spectral-entropy predictability, plus the Friedman/MCB/sign/Wilcoxon statistical tests), `pipeline/clustering.py::clustering_spread` (agreement across repeated clusterings via adjusted Rand index), `pipeline/reconciliation.py` (MinT hierarchical reconciliation with Schafer-Strimmer shrinkage, no model/Excel coupling).

`groups → clusters → (target + sources)` is the core hierarchy. `KabkoData` (`data/kabko.py`) carries everything for one city: raw/data frames, split indices, scalers, datasets, dataloaders, and its trained `model`.

## Model architecture (joint learning)

Everything hangs off `model/modules/main.py::SingleModel`, an encoder-decoder that runs **parallel private and shared branches** — this duality is the whole point of the joint-learning design:

- `PastModel` — encodes the past window (Conv1d `RepresentationModel` → LSTM-style `PastHead`) into hidden state, split into private + shared halves.
- future decoder loop — autoregressive over `future_length` steps using `LILSTMCell2`, with teacher forcing (or an opt-in scheduled-sampling ratio decayed via `ModelUtil.teacher_forcing_ratio_schedule`, see `SingleModel.set_teacher_forcing_ratio`) during training, plus exogenous future features (`FUTURE_EXO_COLS`: holiday/lockdown date flags). `SingleModel(..., direct_multi_horizon=True)` swaps this whole loop for `DirectFutureHead`, a non-autoregressive alternative predicting every future step in one batched op (default `False` keeps the recursive loop).
- `PostFutureModel` / `CombineHead` — merge private + shared outputs into predicted SIRD vars.

Every block exposes `freeze_shared()` / `freeze_private()` for explicit branch-isolation experiments; the default training loop updates both branches. `ModelUtil.alternate_branch_freeze` + `freeze_schedule="alternate"` (opt-in, passed to `eval()`/`make_objective()` in `model/general.py`) alternates which branch is frozen each epoch instead. `SharedMode` and `SourcePick` enums (`model/general.py`) select whether/how sources feed the shared branch.

`model/general.py` is the training driver: `ClusterModel` builds one shared `SingleModel` plus per-kabko models and drives Optuna training; `model/baseline/` and `model/comparison/` hold ablations and comparisons. `model/train.py::eval` is the per-batch train/val/test step (weighted source + target loss, gradient clipping, AMP grad scaler). Model entrypoints default to CPU until `main.init()` is called. `main.py::create_study` supplies the sampler and pruner the objective expects; `main.py::optimize` runs the study sequentially in batches with cache/GC between them; do not set `n_jobs > 1`. The objective reports its running value after each cluster and prunes there, and `find_lr_once` (default on) runs the learning-rate range test once per trial instead of once per cluster. ARIMA's `n_trials` is its total search budget.

Baseline and comparison variants for ablation live in `model/baseline/` (fully_private, fully_shared, no_representation, source_all, source_longest) and `model/comparison/` (arima, sird, arima_sird, naive, theta, linear, gbt) — all comparison arms share `naive.py`'s `fit`/`pred_final`/`eval`/`eval_sample`/`eval_dataset` contract and an `<Name>EvalLog`.

## Conventions

- Stage functions are numbered by execution order and mutate `KabkoData` / `Cluster` objects in place — respect the sequence.
- `data/cols.py` is the single source of truth for column names and the date/exogenous-feature groupings.
- Model blocks take config as **kwargs dicts** (`{}` = build with defaults, `None` = disable that block) — this pattern recurs through `SingleModel` and its submodules.
