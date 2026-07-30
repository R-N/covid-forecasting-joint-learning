# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code (undergrad thesis + published paper) for COVID-19 forecasting of East Java (Jatim) regencies/cities using **joint learning** (multi-task learning with shared + private branches). It is a `pip`-installable Python package, driven from notebooks/scripts that import it — there is no CLI, and `tests/` holds standalone regression checks rather than a test suite. Companion Streamlit web app lives in a separate repo (see README). Current experiment results are not valid evidence of model performance: training initialization, split horizons, baseline compatibility, and statistical testing remain unresolved. Optuna checkpoint scoring, sequential optimization seeds, generic ARIMA sample handling, and clustering/source selection have been corrected. See `INVESTIGATION.md` before reproducing or extending them.

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

The domain model is **SIRD** (Susceptible/Infected/Recovered/Dead compartments). Raw case counts are converted in `pipeline/sird.py` into the SIRD variables the network actually predicts: `SIRD_VARS = [beta, gamma, delta]` (`data/cols.py`). The network outputs these vars; `SingleModel.rebuild` + `pipeline/sird.py` reconstruct absolute IRD counts. All column-name constants live in `data/cols.py` — reuse them, don't hardcode strings.

## Pipeline (the spine of the project)

`pipeline/main.py` exposes **numbered stage functions** run in order; `main.py::main_0`/`main_1` orchestrate them. The numbering *is* the execution order:

- `preprocessing_0` — global-timeseries preparation
- `get_kabkos` / `preprocessing_1` — build per-kabko `KabkoData`, compute SIRD vars
- `preprocessing_2` — **split into groups** by series length (`limit_length`, `limit_date`) and compute train/val/test `split_indices`
- `preprocessing_3` — fit per-group training-only scalers; `main_0` also scales non-date global features from the training portion
- `clustering_1` — **DTW time-series K-Means** (tslearn) on each group's training portion → `Cluster`s. Single-member clusters are dropped as outliers; the **shortest training** series in each cluster becomes the `target`, the rest are `sources`.
- `preprocessing_4` — per-cluster rescale aligned to target
- `preprocessing_5/6/7` — build sliding-window datasets, wrap in torch `DataLoader`s (5/6 = train, 7 = future prediction)

`groups → clusters → (target + sources)` is the core hierarchy. `KabkoData` (`data/kabko.py`) carries everything for one city: raw/data frames, split indices, scalers, datasets, dataloaders, and its trained `model`.

## Model architecture (joint learning)

Everything hangs off `model/modules/main.py::SingleModel`, an encoder-decoder that runs **parallel private and shared branches** — this duality is the whole point of the joint-learning design:

- `PastModel` — encodes the past window (Conv1d `RepresentationModel` → LSTM-style `PastHead`) into hidden state, split into private + shared halves.
- future decoder loop — autoregressive over `future_length` steps using `LILSTMCell2`, with teacher forcing during training and exogenous future features (`FUTURE_EXO_COLS`: holiday/lockdown date flags).
- `PostFutureModel` / `CombineHead` — merge private + shared outputs into predicted SIRD vars.

Every block exposes `freeze_shared()` / `freeze_private()` for explicit branch-isolation experiments; the default training loop updates both branches. `SharedMode` and `SourcePick` enums (`model/general.py`) select whether/how sources feed the shared branch.

`model/general.py` is the training driver: `ClusterModel` builds one shared `SingleModel` plus per-kabko models and drives Optuna training; `model/baseline/` and `model/comparison/` hold ablations and comparisons. `model/train.py::eval` is the per-batch train/val/test step (weighted source + target loss, gradient clipping, AMP grad scaler). Model entrypoints default to CPU until `main.init()` is called. `main.py::optimize` runs the study sequentially in batches with cache/GC between them; do not set `n_jobs > 1`. ARIMA's `n_trials` is its total search budget.

Baseline and comparison variants for ablation live in `model/baseline/` (fully_private, fully_shared, no_representation, source_all, source_longest) and `model/comparison/` (arima, sird, arima_sird).

## Conventions

- Stage functions are numbered by execution order and mutate `KabkoData` / `Cluster` objects in place — respect the sequence.
- `data/cols.py` is the single source of truth for column names and the date/exogenous-feature groupings.
- Model blocks take config as **kwargs dicts** (`{}` = build with defaults, `None` = disable that block) — this pattern recurs through `SingleModel` and its submodules.
