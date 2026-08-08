# covid-forecasting-joint-learning
COVID-19 forecasting model for East Java cities using Joint Learning 

> **Research status:** the committed joint-learning experiment was produced with training and evaluation defects. `INVESTIGATION.md`'s "Must do" rerun-fix list, all 8 "Quick wins", and 4 of its 6 "Big wins" are now applied at the code level (each `py_compile`-checked and covered by a `tests/` regression script) — but none of it has been run against real data, since no environment used to make these changes has torch with CUDA. Treat the fixes as unverified until an actual GPU rerun confirms them. See [INVESTIGATION.md](INVESTIGATION.md) before reproducing or extending experiments.

GPU experiment environment: Python 3.8, CUDA 11.1, and the pinned dependencies in `requirements-experiment.txt`.

The latest revisions align Optuna scoring with restored checkpoints, make clustering/source selection reproducible, correct kabko exclusion plus generic ARIMA sample handling, and default model execution to CPU when initialization is skipped. They also bound the SIRD reconstruction so no rebuilt series has a negative compartment or a falling cumulative count, and cut tuning cost by pruning hopeless Optuna trials and reusing one learning-rate range test per trial. On top of that, split-horizon, metric-selection, and statistical-testing defects are fixed; naive/Theta/linear/GBT baselines, scheduled sampling, a reconstructed-IRD training loss, cluster-target rotation, an alternating branch-freeze schedule, and a non-autoregressive direct multi-horizon decoder head are all now available (most opt-in, defaults unchanged). `INVESTIGATION.md` records what's still open: HFTA member batching and an external EpiCastBench generalisation check are blocked on hardware/data this environment doesn't have. Its `Recommendations` section consolidates all of this into what must be fixed for the rerun to be interpretable, what is cheap, and what is worth real effort, over a literature review that grades each candidate improvement by how well its evidence transfers to this setting and records the ones checked and rejected.

Undergrad thesis: https://digilib.uinsby.ac.id/52500/

Paper: https://journal.maranatha.edu/index.php/jutisi/article/view/4469

Web app (Straemlit): https://github.com/R-N/covid-forecasting-joint-learning-app
