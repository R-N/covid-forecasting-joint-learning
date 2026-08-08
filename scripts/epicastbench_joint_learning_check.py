"""Real joint-learning check: trains `SingleModel`'s actual private+shared
architecture (not just the lightweight comparison baselines in
`epicastbench_check.py`) on real EpiCastBench data, and compares its
forecast RMSSE to naive/linear/theta/gbt.

Honest scope: EpiCastBench provides only aggregate case counts (no R/D
breakdown, no exogenous calendar features), so this does NOT run the
SIRD-rate-prediction + `sird.rebuild` path -- that would need fabricating
R/D data this repo's SIRD modelling doesn't apply to a dataset that
doesn't have it. Instead this tests the *architectural* claim genuinely: a
joint private+shared multi-task encoder-decoder trained across multiple
related regional series, exactly EpiCastBench's own framing ("Recent
advances in multivariate forecasting models better capture complex
temporal dependencies than conventional univariate approaches"). Target:
daily new confirmed cases (first difference of the cumulative series, per
region), single channel, `output_size=1`, no SIRD reconstruction.

Uses `SingleModel(..., direct_multi_horizon=True)` directly (not the full
`ClusterModel`/pipeline machinery, which needs the Excel/DataCenter schema
this dataset doesn't have) -- the shared branch is built by constructing a
template model and reassigning `direct_shared_head`/`shared_head_future_
cell`/`past_model.shared_head` by reference onto every member, the same
sharing mechanism `ClusterModel.__init__` now uses since the bugfix
documented in `tests/test_direct_head_sharing.py`.

Run with:

    python scripts/epicastbench_check.py            # populates the data cache first
    python scripts/epicastbench_joint_learning_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covid_forecasting_joint_learning.model.modules.main import SingleModel
from covid_forecasting_joint_learning.model.loss import RMSSELoss
from covid_forecasting_joint_learning.model import loss_common
from scripts.epicastbench_check import CACHE_DIR, ensure_data, INPUT_CHUNK, OUTPUT_CHUNK

PRIVATE_STATE_SIZE = 8
SHARED_STATE_SIZE = 6
EPOCHS = 150
LR = 1e-3
BATCH_PER_STEP = 8

MEMBERS = ["Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Kerala"]


def load_daily_new_cases(column):
    df = pd.read_csv(CACHE_DIR / "covid_india.csv")
    cumulative = df[column].to_numpy(dtype=float)
    daily = np.diff(cumulative, prepend=cumulative[0])
    return np.clip(daily, 0.0, None)


def make_windows(series, input_chunk, output_chunk):
    """All overlapping (past, future) windows except the last (held out
    as the fixed-origin test window, matching epicastbench_check.py)."""
    n = len(series)
    last_start = n - input_chunk - output_chunk
    windows = [
        (series[start:start + input_chunk], series[start + input_chunk:start + input_chunk + output_chunk])
        for start in range(last_start)
    ]
    test = (series[last_start:last_start + input_chunk], series[last_start + input_chunk:last_start + input_chunk + output_chunk])
    return windows, test


def make_member_model(seed):
    torch.manual_seed(seed)
    return SingleModel(
        input_size_past=1, hidden_size_past=0,
        input_size_future=1, hidden_size_future=0,
        private_state_size=PRIVATE_STATE_SIZE, shared_state_size=SHARED_STATE_SIZE,
        output_size=1, seed_length=5, future_length=OUTPUT_CHUNK,
        past_model={"representation_model": None, "private_head": {}, "shared_head": {}},
        representation_future_model=None,
        private_head_future_cell={}, shared_head_future_cell={},
        post_future_model={}, teacher_forcing=True, use_exo=False, update_hx=True,
        direct_multi_horizon=True,
    )


def main():
    ensure_data()
    torch.manual_seed(0)

    member_data = {name: load_daily_new_cases(name) for name in MEMBERS}
    scales = {name: max(1.0, series.max()) for name, series in member_data.items()}

    train_windows, test_windows = {}, {}
    for name, series in member_data.items():
        windows, test = make_windows(series / scales[name], INPUT_CHUNK, OUTPUT_CHUNK)
        train_windows[name] = windows
        test_windows[name] = test
        print(f"{name}: {len(windows)} training windows, series length {len(series)}")

    # Shared branch: a template model provides the shared submodules; every
    # member gets its own private branch but the SAME shared-branch object
    # references, so gradients from every member accumulate into one set
    # of shared weights (the mechanism ClusterModel.__init__ now uses too).
    shared_template = make_member_model(seed=0)
    members = {}
    for i, name in enumerate(MEMBERS):
        m = make_member_model(seed=100 + i)
        m.direct_shared_head = shared_template.direct_shared_head
        m.shared_head_future_cell = shared_template.shared_head_future_cell
        m.past_model.shared_head = shared_template.past_model.shared_head
        members[name] = m

    all_models = list(members.values())
    params = list({id(p): p for m in all_models for p in m.parameters()}.values())
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_fn = RMSSELoss(reduction="sum")

    for model in all_models:
        model.train()

    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        total_loss = torch.zeros(())
        for name in MEMBERS:
            windows = train_windows[name]
            idx = rng.integers(0, len(windows), size=min(BATCH_PER_STEP, len(windows)))
            pasts = torch.tensor(np.stack([windows[i][0] for i in idx]), dtype=torch.float32).unsqueeze(-1)
            futures = torch.tensor(np.stack([windows[i][1] for i in idx]), dtype=torch.float32).unsqueeze(-1)
            dummy_seed = torch.zeros(pasts.shape[0], 5, 1)

            pred = members[name](pasts, dummy_seed, future=None, future_exo=None)
            total_loss = total_loss + loss_fn(pasts, futures, pred) / pasts.shape[0]

        total_loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch}: total_loss={total_loss.item():.4f}")

    for model in all_models:
        model.eval()

    print("\n--- Held-out forecast RMSSE (real counts, joint-learning SingleModel) ---")
    results = {}
    with torch.no_grad():
        for name in MEMBERS:
            test_past, test_future = test_windows[name]
            past_t = torch.tensor(test_past, dtype=torch.float32).view(1, INPUT_CHUNK, 1)
            dummy_seed = torch.zeros(1, 5, 1)
            pred = members[name](past_t, dummy_seed, future=None, future_exo=None)
            pred = pred.squeeze(0).squeeze(-1).numpy() * scales[name]
            true_future = test_future * scales[name]
            true_past = test_past * scales[name]

            rmsse = loss_common.rmsse(true_past.reshape(-1, 1), true_future.reshape(-1, 1), pred.reshape(-1, 1))[0]
            results[name] = {"joint_learning": float(rmsse)}
            print(f"{name}: joint_learning_rmsse={rmsse:.4f}")

    print("\n--- Same target, same split: naive/linear/theta/gbt baselines ---")
    from covid_forecasting_joint_learning.model.comparison.linear import LinearModel
    from covid_forecasting_joint_learning.model.comparison.theta import ThetaModel
    from covid_forecasting_joint_learning.model.comparison.gbt import GBTModel

    for name in MEMBERS:
        test_past, test_future = test_windows[name]
        seed = test_past * scales[name]
        test = test_future * scales[name]
        seed_3col = np.tile(seed.reshape(-1, 1), (1, 3))

        naive_pred = np.full(OUTPUT_CHUNK, seed[-1])
        linear_model = LinearModel()
        linear_model.fit(seed_3col)
        linear_pred = linear_model.pred_final(OUTPUT_CHUNK)[:, 0]
        theta_model = ThetaModel()
        theta_model.fit(seed_3col)
        theta_pred = theta_model.pred_final(OUTPUT_CHUNK)[:, 0]
        gbt_model = GBTModel()
        gbt_model.fit(seed_3col)
        gbt_pred = gbt_model.pred_final(OUTPUT_CHUNK)[:, 0]

        def score(pred):
            return float(loss_common.rmsse(seed.reshape(-1, 1), test.reshape(-1, 1), np.asarray(pred).reshape(-1, 1))[0])

        results[name].update({
            "naive": score(naive_pred),
            "linear": score(linear_pred),
            "theta": score(theta_pred),
            "gbt": score(gbt_pred),
        })
        print(f"{name}: {results[name]}")

    return results


if __name__ == "__main__":
    main()
