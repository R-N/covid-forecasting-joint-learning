# covid-forecasting-joint-learning
COVID-19 forecasting model for East Java cities using Joint Learning 

> **Research status:** the committed joint-learning experiment was produced with training and evaluation defects. Remaining training initialization, split-horizon, baseline, metric-selection, and statistical issues prevent a valid comparison. See [INVESTIGATION.md](INVESTIGATION.md) before reproducing or extending experiments.

GPU experiment environment: Python 3.8, CUDA 11.1, and the pinned dependencies in `requirements-experiment.txt`.

The latest revisions align Optuna scoring with restored checkpoints, make clustering/source selection reproducible, correct kabko exclusion plus generic ARIMA sample handling, and default model execution to CPU when initialization is skipped.

Undergrad thesis: https://digilib.uinsby.ac.id/52500/

Paper: https://journal.maranatha.edu/index.php/jutisi/article/view/4469

Web app (Straemlit): https://github.com/R-N/covid-forecasting-joint-learning-app
