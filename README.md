# covid-forecasting-joint-learning
COVID-19 forecasting model for East Java cities using Joint Learning 

> **Research status:** the committed joint-learning experiment was produced with training and evaluation defects. Remaining split-horizon, metric-selection, baseline-compatibility, and statistical issues prevent a valid comparison. See [INVESTIGATION.md](INVESTIGATION.md) before reproducing or extending experiments.

GPU experiment environment: Python 3.8, CUDA 11.1, and the pinned dependencies in `requirements-experiment.txt`.

The latest revisions align Optuna scoring with restored checkpoints, make clustering/source selection reproducible, correct kabko exclusion plus generic ARIMA sample handling, and default model execution to CPU when initialization is skipped. They also bound the SIRD reconstruction so no rebuilt series has a negative compartment or a falling cumulative count, and cut tuning cost by pruning hopeless Optuna trials and reusing one learning-rate range test per trial. `INVESTIGATION.md` records the remaining accuracy and cost work identified for the rerun. Its `Recommendations` section consolidates that into what must be fixed for the rerun to be interpretable, what is cheap, and what is worth real effort, over a literature review that grades each candidate improvement by how well its evidence transfers to this setting and records the ones checked and rejected.

Undergrad thesis: https://digilib.uinsby.ac.id/52500/

Paper: https://journal.maranatha.edu/index.php/jutisi/article/view/4469

Web app (Straemlit): https://github.com/R-N/covid-forecasting-joint-learning-app
