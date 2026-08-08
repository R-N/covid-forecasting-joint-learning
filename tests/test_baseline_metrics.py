"""Regression check for the ARIMA/SIRD baseline usability fixes.

Two defects: SIRDModel/ARIMASIRDModel's single-sample `eval()`/`test()`
defaulted to the feature-summed scalar `rmsse` (via `wrap_reduce`'s default
`reduce_feature=True`), while `SIRDEvalLog.log()`/`ARIMASIRDEvalLog.log()`
assert `len(loss) == 3` (one value per I/R/D column) -- a scalar has no
`len()`. And `ARIMASIRDModel.eval_sample()` unpacked the standard
`label_dataset_0` 8-field sample as if it were a bare 4-tuple, handing the
wide multi-feature `past` array (and the rate-space `future`) to fields
that need the 3-column rate history and the S/I/R/D seed instead.

statsmodels/optuna/xlrd/lmfit are stubbed; none of their real behavior is
exercised by these checks. Run with:

    python tests/test_baseline_metrics.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "statsmodels" not in sys.modules:
        statsmodels = types.ModuleType("statsmodels")
        tsa = types.ModuleType("statsmodels.tsa")
        statespace = types.ModuleType("statsmodels.tsa.statespace")
        sarimax = types.ModuleType("statsmodels.tsa.statespace.sarimax")
        sarimax.SARIMAX = type("SARIMAX", (), {})
        statespace.sarimax = sarimax
        tsa.statespace = statespace
        statsmodels.tsa = tsa
        for name, mod in [
            ("statsmodels", statsmodels),
            ("statsmodels.tsa", tsa),
            ("statsmodels.tsa.statespace", statespace),
            ("statsmodels.tsa.statespace.sarimax", sarimax),
        ]:
            sys.modules[name] = mod
    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        optuna.create_study = lambda *a, **k: None
        samplers = types.ModuleType("optuna.samplers")
        samplers.TPESampler = type("TPESampler", (), {})
        structs = types.ModuleType("optuna.structs")
        structs.TrialPruned = type("TrialPruned", (Exception,), {})
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.TrialState = type("TrialState", (), {})
        optuna.samplers = samplers
        optuna.structs = structs
        optuna.trial = trial_mod
        for name, mod in [
            ("optuna", optuna),
            ("optuna.samplers", samplers),
            ("optuna.structs", structs),
            ("optuna.trial", trial_mod),
        ]:
            sys.modules[name] = mod
    if "xlrd" not in sys.modules:
        xlrd = types.ModuleType("xlrd")
        xlrd.XLRDError = type("XLRDError", (Exception,), {})
        sys.modules["xlrd"] = xlrd
    if "lmfit" not in sys.modules:
        lmfit = types.ModuleType("lmfit")
        lmfit.minimize = lambda *a, **k: None
        lmfit.Parameters = type("Parameters", (), {})
        sys.modules["lmfit"] = lmfit
    if "line_profiler" not in sys.modules:
        line_profiler = types.ModuleType("line_profiler")
        line_profiler.LineProfiler = type("LineProfiler", (), {})
        sys.modules["line_profiler"] = line_profiler


def test_sird_model_single_eval_defaults_to_per_ird():
    from covid_forecasting_joint_learning.model.comparison import sird as sird_cmp

    # rmsse_per_ird must stay unreduced on the feature axis (shape (3,) for
    # a single sample), unlike the scalar rmsse used for search/comparison.
    past = np.ones((10, 3)) * 2.0
    future = np.ones((5, 3)) * 3.0
    pred = np.ones((5, 3)) * 3.5

    per_ird = sird_cmp.rmsse_per_ird(past, future, pred)
    assert per_ird.shape == (3,), per_ird.shape

    scalar = sird_cmp.rmsse(past, future, pred)
    assert np.ndim(scalar) == 0, scalar

    # The class must default to the per-IRD variant so a direct eval() call
    # satisfies SIRDEvalLog.log()'s `assert len(loss) == 3`.
    assert sird_cmp.SIRDModel.__init__.__defaults__[0] is sird_cmp.rmsse_per_ird
    import inspect
    assert inspect.signature(sird_cmp.SIRDModel.eval).parameters["loss_fn"].default is sird_cmp.rmsse_per_ird

    # eval_dataset (used by search_greedy/search_optuna, which compare with
    # `loss < best_loss`) must keep the scalar default.
    assert inspect.signature(sird_cmp.SIRDModel.eval_dataset).parameters["loss_fn"].default is sird_cmp.rmsse


def test_arima_sird_eval_sample_unpacks_standard_dataset():
    from covid_forecasting_joint_learning.model.comparison import arima_sird

    model = arima_sird.ARIMASIRDModel(models=[None, None, None], population=1_000_000)

    captured = {}

    def fake_eval(self, past, final_seed, future_final, exo=None, past_exo=None, loss_fn=None):
        captured.update(past=past, final_seed=final_seed, future_final=future_final, exo=exo, past_exo=past_exo)
        return np.array([0.1, 0.2, 0.3])

    model.eval = types.MethodType(fake_eval, model)

    # Mirrors label_dataset_0's 8-field tuple: (past, past_seed, past_exo,
    # future, future_exo, final_seed, future_final, index).
    past = np.zeros((30, 12))          # wide multi-feature past window
    past_seed = np.ones((5, 3)) * 1.0  # beta/gamma/delta history
    past_exo = np.ones((5, 3)) * 2.0
    future = np.ones((14, 3)) * 3.0    # rate-space targets, not used by ARIMA-SIRD
    future_exo = np.ones((14, 3)) * 4.0
    final_seed = np.ones((5, 4)) * 5.0  # S/I/R/D seed
    future_final = np.ones((14, 3)) * 6.0  # IRD ground truth
    index = list(range(14))
    sample = (past, past_seed, past_exo, future, future_exo, final_seed, future_final, index)

    model.eval_sample(sample, use_exo=True)

    assert captured["past"] is past_seed, "must fit on the 3-column rate history, not the wide past window"
    assert captured["final_seed"] is final_seed
    assert captured["future_final"] is future_final
    assert captured["exo"] is future_exo
    assert captured["past_exo"] is past_exo

    # use_exo=False must drop the exogenous arrays.
    model.eval_sample(sample, use_exo=False)
    assert captured["exo"] is None
    assert captured["past_exo"] is None

    # A 7-field sample (no trailing index) must unpack identically.
    model.eval_sample(sample[:7], use_exo=True)
    assert captured["past"] is past_seed
    assert captured["future_final"] is future_final


if __name__ == "__main__":
    install_stubs()
    test_sird_model_single_eval_defaults_to_per_ird()
    test_arima_sird_eval_sample_unpacks_standard_dataset()
    print("ok")
