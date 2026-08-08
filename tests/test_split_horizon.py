"""Regression check for the calc_split forecast horizon.

`calc_split` computed its train/val/test boundaries assuming a zero-length
forecast horizon (`future_size` defaulted to 0 and was never threaded through
`preprocessing_2`), while every downstream window builder
(`preprocessing_5`/`slice_dataset`) uses a 14-day horizon. That mismatch left
the val/test portions with fewer valid forecast windows than the requested
portion implies -- in the extreme, zero. This pins the invariant that fixes
it: the number of sliding windows `slice_dataset` can build in a segment
(`segment_len - future_size + 1` at stride 1) must equal the window count
`calc_split` allocated to that segment.

sklearn and tslearn are stubbed so this runs without those (heavy, GPU-era)
dependencies; `calc_split` never touches either. Run with:

    python tests/test_split_horizon.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_stubs():
    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        preprocessing = types.ModuleType("sklearn.preprocessing")
        preprocessing.MinMaxScaler = type("MinMaxScaler", (), {})
        preprocessing.StandardScaler = type("StandardScaler", (), {})
        sklearn.preprocessing = preprocessing
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.preprocessing"] = preprocessing
    if "tslearn" not in sys.modules:
        tslearn = types.ModuleType("tslearn")
        utils = types.ModuleType("tslearn.utils")
        utils.to_time_series_dataset = lambda *a, **k: None
        clustering = types.ModuleType("tslearn.clustering")
        clustering.TimeSeriesKMeans = type("TimeSeriesKMeans", (), {})
        clustering.silhouette_score = lambda *a, **k: None
        metrics = types.ModuleType("tslearn.metrics")
        metrics.dtw = lambda *a, **k: None
        tslearn.utils = utils
        tslearn.clustering = clustering
        tslearn.metrics = metrics
        sys.modules["tslearn"] = tslearn
        sys.modules["tslearn.utils"] = utils
        sys.modules["tslearn.clustering"] = clustering
        sys.modules["tslearn.metrics"] = metrics
    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        trial_mod = types.ModuleType("optuna.trial")
        trial_mod.TrialState = type("TrialState", (), {})
        optuna.trial = trial_mod
        sys.modules["optuna"] = optuna
        sys.modules["optuna.trial"] = trial_mod


def windows_allocated(n_rows, val_portion=0.25, test_portion=0.25, past_size=30, future_size=14):
    n = n_rows - past_size - future_size + 1
    val_len, test_len = int(val_portion * n), int(test_portion * n)
    train_len = n - (val_len + test_len)
    return train_len, val_len, test_len


def test_default_horizon_reserves_room_for_every_allocated_window():
    import pandas as pd
    from covid_forecasting_joint_learning.pipeline.preprocessing import calc_split

    n_rows = 200
    past_size, future_size = 30, 14
    df = pd.DataFrame({"x": range(n_rows)})  # RangeIndex: label == position

    train_end, val_start, val_end, test_start = calc_split(
        df, past_size=past_size, future_size=future_size
    )
    train_len, val_len, test_len = windows_allocated(
        n_rows, past_size=past_size, future_size=future_size
    )

    # slice_dataset's window count at stride 1: segment_len - future_size + 1.
    # Only the tail (test) segment runs to len(df) with no further split
    # boundary after it, so it gets exactly its allocated window count.
    # Internal boundaries (train|val, val|test) each forfeit future_size - 1
    # windows to the horizon of the segment that follows -- inherent to
    # horizon-respecting contiguous splits, not something calc_split can
    # avoid.
    val_windows = (test_start - val_start) - future_size + 1
    test_windows = (n_rows - test_start) - future_size + 1
    assert val_windows == val_len - (future_size - 1), (val_windows, val_len)
    assert test_windows == test_len, (test_windows, test_len)


def test_zero_horizon_starves_the_tail_segments():
    import pandas as pd
    from covid_forecasting_joint_learning.pipeline.preprocessing import calc_split

    # Pins the bug being fixed: computing boundaries with future_size=0 while
    # windows are actually built with future_size=14 shorts the test segment
    # by (new_future_size - 1) rows relative to what calc_split allocated.
    n_rows = 200
    past_size, future_size = 30, 14
    df = pd.DataFrame({"x": range(n_rows)})

    _, _, _, test_start_bug = calc_split(df, past_size=past_size, future_size=0)
    _, val_len_bug, test_len_bug = windows_allocated(n_rows, past_size=past_size, future_size=0)

    test_windows_bug = (n_rows - test_start_bug) - future_size + 1
    assert test_windows_bug == test_len_bug - future_size, (test_windows_bug, test_len_bug)
    assert test_windows_bug < test_len_bug


def test_preprocessing_2_threads_future_size_to_calc_split():
    # pipeline.main pulls in torch/line_profiler/pydrive2 transitively; a
    # static check of the default value avoids stubbing that whole chain
    # for a single-parameter assertion.
    import ast

    source = Path(__file__).resolve().parents[1] / "covid_forecasting_joint_learning" / "pipeline" / "main.py"
    tree = ast.parse(source.read_text())
    names = {"preprocessing_2", "_preprocessing_2", "__preprocessing_2"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.add(node.name)
            defaults = {
                arg.arg: default
                for arg, default in zip(reversed(node.args.args), reversed(node.args.defaults))
            }
            future_size_default = defaults["future_size"]
            assert isinstance(future_size_default, ast.Constant) and future_size_default.value == 14, node.name
    assert found == names, found


if __name__ == "__main__":
    install_stubs()
    test_default_horizon_reserves_room_for_every_allocated_window()
    test_zero_horizon_starves_the_tail_segments()
    test_preprocessing_2_threads_future_size_to_calc_split()
    print("ok")
