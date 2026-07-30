# Agent Guide

## Workflow

- This is an import-driven Python package, not a CLI application. There is no linter, formatter, CI workflow, or lockfile. Use `pip install -e .` for development; package with `python setup.py sdist bdist_wheel`.
- `tests/` holds standalone regression checks, not a suite: plain `assert`, heavy dependencies stubbed, run one at a time with `python tests/<name>.py`. Match that pattern; there is no runner to register with.
- For reproducible GPU experiments, use Python 3.8 with CUDA 11.1 and run `pip install -r requirements-experiment.txt` before `pip install -e .`. `pipeline/eval.py` requires `orange3`.
- For an executable smoke check beyond packaging, install editable and import the package or the changed module, or run the relevant script in `tests/`. Do not claim a test suite was run.
- `covid_forecasting_joint_learning.main.init()` sets PyTorch's global default tensor type and selects CUDA only when available; call it deliberately in notebooks/scripts.
- Model entrypoints default to CPU before `main.init()`; use `main.init()` deliberately when GPU execution or its global tensor-type side effect is required.
- Current joint-learning results are not valid evidence of model performance. `INVESTIGATION.md` tracks remaining training, split-horizon, metric-selection, baseline, and statistical blockers, plus an Improvement Opportunities section covering accuracy and training/tuning cost work; consult it before reproducing or extending experiments.
- Recent fixes align Optuna scores with restored checkpoints, adapt standard exogenous samples for ARIMA, record distinct clustering seeds, and seed sequential neural/baseline optimization. Parallel Optuna trials are intentionally unsupported because model RNG state is process-global.

## Data And Pipeline

- `DataCenter.load_excel()` requires workbook sheets named `covid_indo`, `covid_jatim`, `vaccine`, `test`, `population`, `psbb`, `ppkm`, `ppkm_mikro`, `long_holiday`, `pilkada`, and `other_dates`.
- Use `main.main_0()` / `main.main_1()` or preserve the numbered order in `pipeline/main.py`. Stages mutate `KabkoData` and `Cluster` state in place; later stages require prior scalers, split indices, clustering, and aligned data.
- Fit imputation, scalers, and DTW clustering from training observations only. The core hierarchy is `groups -> clusters -> target + sources`; single-member clusters are dropped, each cluster targets its shortest training series, and `preprocessing_4` aligns members to that target.
- Networks predict SIRD rates (`beta`, `gamma`, `delta`), not absolute cases. Rebuild IRD counts through the SIRD pipeline/model path. Reuse names and feature groups from `data/cols.py`; do not hardcode column strings.

## Model Changes

- `model/general.py` builds a shared `SingleModel` plus per-kabko models and drives Optuna training; ablations belong in `model/baseline/` and statistical comparisons in `model/comparison/`.
- Model-block keyword configuration is semantic: `{}` builds a block with defaults, while `None` disables it. Keep that distinction when changing architecture defaults.
- The joint-learning model has private and shared branches. Preserve the paired `freeze_shared()` / `freeze_private()` behavior and source-selection modes when changing training or model wiring.
