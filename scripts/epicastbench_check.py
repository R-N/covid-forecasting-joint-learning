"""External generalisation check against EpiCastBench (arXiv:2605.11598),
a public 40-dataset multivariate epidemic-forecasting benchmark
(code: https://github.com/aimltsf/EpiCastBench, data: Kaggle). This is
INVESTIGATION.md's Big wins #6: "The only way to show a method generalises
beyond one province of one country in one epidemic, and its baselines are
already implemented."

Scope: this repo's comparison baselines only (naive/linear/theta/gbt from
model/comparison/), scored on the single case-count channel per region --
NOT the full SingleModel joint-learning architecture, which needs the SIRD
R/D breakdown and exogenous calendar features EpiCastBench's plain
case-count CSVs don't provide, and not the full 40-dataset/15-model sweep
(that needs EpiCastBench's own GPU-hungry deep baselines and real compute
budget). This proves the code path generalizes; it is not a claim of
state-of-the-art performance on EpiCastBench.

Protocol matches EpiCastBench's own Code/config.py: INPUT_CHUNK=24 past
steps, OUTPUT_CHUNK=12 future steps, fixed-origin (the last OUTPUT_CHUNK
points held out as test, scored with this repo's own naive-scaled RMSSE
(`model/loss_common.py::rmsse`) rather than EpiCastBench's own MASE/SMAPE,
so it is directly comparable to every other comparison arm in this repo).

`LinearModel.fit`/`ThetaModel.fit`/`GBTModel.fit` all require a 3-column
I/R/D input; satisfied honestly here by triplicating the real
single-column series (not fake zero-padding) and reading only column 0's
result -- each column fits an independent model, so columns 1-2 being
present doesn't affect column 0's fit or forecast.

Downloads and caches the dataset zip from EpiCastBench's public Kaggle
release on first run (no Kaggle API key needed -- confirmed reachable over
plain HTTPS). Run with:

    python scripts/epicastbench_check.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covid_forecasting_joint_learning.model import loss_common
from covid_forecasting_joint_learning.model.comparison.linear import LinearModel
from covid_forecasting_joint_learning.model.comparison.theta import ThetaModel
from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/aimltsf/epicastbench"
CACHE_DIR = Path(__file__).resolve().parent / ".epicastbench_cache"
INPUT_CHUNK = 24
OUTPUT_CHUNK = 12

# One representative series per continent/disease -- not the full 40-dataset
# sweep (see module docstring). Enough to prove the baselines run correctly,
# with no shape/schema surprises, on genuinely external data.
SERIES = [
    ("covid_india.csv", "Delhi"),
    ("covid_us.csv", "California"),
    ("dengue_brazil.csv", "Bahia"),
]


def ensure_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if any(CACHE_DIR.glob("*.csv")):
        return
    print(f"Downloading EpiCastBench dataset to {CACHE_DIR} ...")
    with urllib.request.urlopen(KAGGLE_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(CACHE_DIR)


def load_column(csv_name, column):
    df = pd.read_csv(CACHE_DIR / csv_name)
    return df[column].to_numpy(dtype=float)


def split(series):
    assert len(series) >= INPUT_CHUNK + OUTPUT_CHUNK, "series too short for the protocol window"
    test = series[-OUTPUT_CHUNK:]
    seed = series[-(OUTPUT_CHUNK + INPUT_CHUNK):-OUTPUT_CHUNK]
    return seed, test


def naive_forecast(seed, horizon):
    return np.full(horizon, seed[-1])


def score(seed, test, pred):
    past = seed.reshape(-1, 1)
    future = test.reshape(-1, 1)
    pred = np.asarray(pred).reshape(-1, 1)
    return float(loss_common.rmsse(past, future, pred)[0])


def run(csv_name, column):
    series = load_column(csv_name, column)
    seed, test = split(series)
    seed_3col = np.tile(seed.reshape(-1, 1), (1, 3))

    naive_pred = naive_forecast(seed, OUTPUT_CHUNK)

    linear_model = LinearModel()
    linear_model.fit(seed_3col)
    linear_pred = linear_model.pred_final(OUTPUT_CHUNK)[:, 0]

    theta_model = ThetaModel()
    theta_model.fit(seed_3col)
    theta_pred = theta_model.pred_final(OUTPUT_CHUNK)[:, 0]

    gbt_model = GBTModel()
    gbt_model.fit(seed_3col)
    gbt_pred = gbt_model.pred_final(OUTPUT_CHUNK)[:, 0]

    return {
        "series": f"{csv_name}:{column}",
        "naive_rmsse": score(seed, test, naive_pred),
        "linear_rmsse": score(seed, test, linear_pred),
        "theta_rmsse": score(seed, test, theta_pred),
        "gbt_rmsse": score(seed, test, gbt_pred),
    }


def main():
    ensure_data()
    results = [run(csv_name, column) for csv_name, column in SERIES]
    for row in results:
        print(row)
    return results


if __name__ == "__main__":
    main()
