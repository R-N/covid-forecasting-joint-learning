# Agent Guide

## Workflow

- This is an import-driven Python package, not a CLI application. There are no committed tests, linter, formatter, CI workflow, or lockfile. Use `pip install -e .` for development; package with `python setup.py sdist bdist_wheel`.
- Dependency compatibility is intentional: `numpy==1.23.1`, `pandas==1.4.3`, and `optuna==2.10.1` are pinned; PyTorch is unpinned, with `1.8.1` noted for GPU use. `pipeline/eval.py` requires `orange3`.
- For an executable smoke check beyond packaging, install editable and import the package or the changed module. Do not claim a test suite was run.
- `covid_forecasting_joint_learning.main.init()` sets PyTorch's global default tensor type and selects CUDA only when available; call it deliberately in notebooks/scripts.
- Current joint-learning results are not valid evidence of model performance. `INVESTIGATION.md` tracks remaining metric-selection, baseline, and statistical blockers; consult it before reproducing or extending experiments.

## Data And Pipeline

- `DataCenter.load_excel()` requires workbook sheets named `covid_indo`, `covid_jatim`, `vaccine`, `test`, `population`, `psbb`, `ppkm`, `ppkm_mikro`, `long_holiday`, `pilkada`, and `other_dates`.
- Use `main.main_0()` / `main.main_1()` or preserve the numbered order in `pipeline/main.py`. Stages mutate `KabkoData` and `Cluster` state in place; later stages require prior scalers, split indices, clustering, and aligned data.
- Fit imputation, scalers, and DTW clustering from training observations only. The core hierarchy is `groups -> clusters -> target + sources`; single-member clusters are dropped, each cluster targets its shortest training series, and `preprocessing_4` aligns members to that target.
- Networks predict SIRD rates (`beta`, `gamma`, `delta`), not absolute cases. Rebuild IRD counts through the SIRD pipeline/model path. Reuse names and feature groups from `data/cols.py`; do not hardcode column strings.

## Model Changes

- `model/general.py` builds a shared `SingleModel` plus per-kabko models and drives Optuna training; ablations belong in `model/baseline/` and statistical comparisons in `model/comparison/`.
- Model-block keyword configuration is semantic: `{}` builds a block with defaults, while `None` disables it. Keep that distinction when changing architecture defaults.
- The joint-learning model has private and shared branches. Preserve the paired `freeze_shared()` / `freeze_private()` behavior and source-selection modes when changing training or model wiring.
