# Agent Guide

## Workflow

- This is an import-driven Python package, not a CLI application. There is no linter, formatter, CI workflow, or lockfile. Use `pip install -e .` for development; package with `python setup.py sdist bdist_wheel`.
- `tests/` holds standalone regression checks, not a suite: plain `assert`, heavy dependencies stubbed, run one at a time with `python tests/<name>.py`. Match that pattern; there is no runner to register with.
- For reproducible GPU experiments, use Python 3.8 with CUDA 11.1 and run `pip install -r requirements-experiment.txt` before `pip install -e .`. `pipeline/eval.py` requires `orange3`.
- For an executable smoke check beyond packaging, install editable and import the package or the changed module, or run the relevant script in `tests/`. Do not claim a test suite was run.
- `covid_forecasting_joint_learning.main.init()` sets PyTorch's global default tensor type and selects CUDA only when available; call it deliberately in notebooks/scripts.
- Model entrypoints default to CPU before `main.init()`; use `main.init()` deliberately when GPU execution or its global tensor-type side effect is required.
- Current joint-learning results are not valid evidence of model performance. `INVESTIGATION.md` tracks remaining training, split-horizon, metric-selection, baseline, and statistical blockers, plus an Improvement Opportunities section covering accuracy and training/tuning cost work; consult it before reproducing or extending experiments.
- Start at the `Recommendations` section of `INVESTIGATION.md`. It is the reading order for everything below it: what must be done for the rerun to be interpretable, what is cheap, what is worth real effort, and what was checked and rejected. The sections after it are organised by when each finding was made, not by priority.
- Improvement Opportunities ends with a literature review that grades each idea on whether its evidence comes from a setting resembling this one (roughly 38 short daily series, 14-day horizon, tiny models, single split). Ideas checked and rejected are recorded with their reasons, so read it before proposing efficient attention, low-rank factorization, gradient surgery, learned losses, active learning, sample-level curricula, generative oversampling, or subsequence-clustering criticisms.
- The review's central external finding: in the US COVID-19 Forecast Hub case-forecast retrospective, only 7 of 22 teams beat a last-value-carried-forward baseline at state level, and skill was worse at county level, which is this project's geographic scale. Treat the naive baseline as the result that decides whether there is a finding, not as a formality.
- That pessimistic number comes from prospective real-time forecasting, where data revisions are unresolved; retrospective benchmarks on finalised data report deep learning comfortably beating naive. This project is retrospective, so the optimistic setting is the comparable one, but any claim it produces is retrospective skill on finalised data and must be stated that way rather than as operational forecasting skill.
- `EpiCastBench` (40 public epidemic datasets, 15 implemented models including Naive, DLinear, LSTM, TSMixer) is the external generalisation check this project cannot produce from 38 kabko alone, and it supplies baselines and metrics already implemented. It uses Friedman with post-hoc Multiple Comparisons with the Best, which is the field convention for the statistical-testing blocker.
- `loss_common.py::naive()` scales `msse`/`rmsse` against first differences (`step=1`), not a seasonal naive. This was checked against the literature and is fine: M5 defined RMSSE the same way on daily series with weekly seasonality, and the denominator is model-independent so it cannot bias a comparison. What does matter is that `rmsse < 1` means beating the in-sample one-step naive, which is not the same claim as beating an out-of-sample 14-step naive, so the explicit naive baseline arm is still required.
- Generic PyTorch tuning advice does not transfer here: whole splits are already CUDA tensors so `num_workers`/`pin_memory` are inert, `cudnn.benchmark` would thrash because batch sizes vary by design, and `torch.compile`/CUDA graphs do not exist in the pinned torch 1.8. This workload is framework-overhead-bound, not FLOP-bound, so anything that only reduces FLOPs (AMP included) will not help.
- Before hand-rolling the member-batching optimisation, look at HFTA (MLSys 2021): it is the same operator-level fusion of architecturally identical models, published as a torch-1.8-era extension library. Cluster members within a trial fuse cleanly; Optuna trials that differ in layer shapes do not. CUDA MPS is the no-code-change fallback, not the better option.
- Epidemic forecastability rises with the population of the target, and kabko populations span roughly an order of magnitude. Aggregate comparisons across kabko can therefore measure population size rather than method. Report spectral-entropy forecastability as a covariate and use log-transformed or relative scoring, rather than comparing raw aggregates.
- Recent fixes align Optuna scores with restored checkpoints, adapt standard exogenous samples for ARIMA, record distinct clustering seeds, and seed sequential neural/baseline optimization. Parallel Optuna trials are intentionally unsupported because model RNG state is process-global.
- `general.eval` and `general.make_objective` default to `find_lr_once=True`, which runs the learning-rate range test once per run or trial and reuses the result across that run's clusters. Keep the two sides equal, or the search and the final fit pick their learning rate differently.
- Build the neural study with `main.create_study()`. The objective reports its running value after every cluster and prunes there, so a study without a pruner silently loses that saving, and pruned trials count against the `main.optimize()` budget.

## Data And Pipeline

- `DataCenter.load_excel()` requires workbook sheets named `covid_indo`, `covid_jatim`, `vaccine`, `test`, `population`, `psbb`, `ppkm`, `ppkm_mikro`, `long_holiday`, `pilkada`, and `other_dates`.
- Use `main.main_0()` / `main.main_1()` or preserve the numbered order in `pipeline/main.py`. Stages mutate `KabkoData` and `Cluster` state in place; later stages require prior scalers, split indices, clustering, and aligned data.
- Fit imputation, scalers, and DTW clustering from training observations only. The core hierarchy is `groups -> clusters -> target + sources`; single-member clusters are dropped, each cluster targets its shortest training series, and `preprocessing_4` aligns members to that target.
- Networks predict SIRD rates (`beta`, `gamma`, `delta`), not absolute cases. Rebuild IRD counts through the SIRD pipeline/model path, which clamps rates and flows so no rebuilt series has a falling cumulative R or D or a negative compartment. Reuse names and feature groups from `data/cols.py`; do not hardcode column strings.

## Model Changes

- `model/general.py` builds a shared `SingleModel` plus per-kabko models and drives Optuna training; ablations belong in `model/baseline/` and statistical comparisons in `model/comparison/`.
- Model-block keyword configuration is semantic: `{}` builds a block with defaults, while `None` disables it. Keep that distinction when changing architecture defaults.
- The joint-learning model has private and shared branches. Preserve the paired `freeze_shared()` / `freeze_private()` behavior and source-selection modes when changing training or model wiring.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
