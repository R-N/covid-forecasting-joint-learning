# Experiment Investigation

## Status

The prior experiment cannot support a conclusion that joint learning does or does not improve forecasting. Core implementation defects were corrected in `96a0104`, but the blockers below must be fixed before a rerun can compare methods. Improvement Opportunities collects the accuracy and cost work that is worth doing alongside that rerun but is not required for correctness.

## Fixes Applied

- The decoder now retains the most recent seed window, allowing predicted or teacher-forced rates and future exogenous features to reach later forecast steps.
- Continued validation reuse has been removed. The default is now to test the validation-selected checkpoint directly.
- Zero filling is causal, global features are scaled per group from training observations only, and local SIRD scaling and clustering use the training split only.
- OneCycleLR now advances for each successful optimizer update and resets after its configured total step count.
- Shared/private freezing uses the correct model instances and branch-specific modules; source loss is normalized so `source_weight` is the combined contribution of all sources.
- Cluster alignment now rejects sources without sufficient history before the target training split.
- CUDA conversion, Friedman statistics, ARIMA-SIRD metric scaling, SIRD forecast timing, and multi-step cumulative reconstruction have been corrected.
- Optuna now scores the checkpoint selected by early stopping; ARIMA has an explicit adapter for standard exogenous samples, and clustering-consistency runs use recorded distinct seeds.
- `requirements-experiment.txt` pins the Python 3.8/CUDA 11.1 experiment environment, including TensorBoard required by `SummaryWriter`.
- Group kabko removal now compares names correctly, `SourcePick.CLOSEST` supplies DTW with SIRD-rate arrays, and baseline Optuna searches use seeded samplers.
- Neural trials use distinct deterministic seeds and `main.optimize()` rejects parallel execution, which would race process-global RNG state.
- Model entrypoints default to CPU when `main.init()` was not called, and ARIMA treats `n_trials` as its total Optuna budget, matching SIRD.
- `main_1()` forwards `limit_data` to `main_0()`, so `limit_data=False` no longer leaves scalers and clustering silently split-limited while `preprocessing_4` honours the caller.
- `EarlyStopping.calculate_interval()` handles the empty first-epoch history by treating the current loss as the whole interval, covered by `tests/test_early_stopping.py`.
- Each member's batch loss is divided by that member's own batch size before weighting, so a member with more data no longer contributes more than `kabko.weight` states, and epoch losses are averaged over the batches actually consumed.
- `pipeline/sird.py::rebuild` bounds reconstruction: rates are clamped at zero, removal from I is capped at I, and inflow to I is capped at S, so no rebuilt series has a falling cumulative R or D, a rising S, or a negative compartment. The clamps are inert for valid rates and leave NaN predictions visible. Covered by `tests/test_sird_rebuild.py`.

## Required Rerun Design

- Add a decoder regression test verifying that each generated output replaces the oldest seed entry.
- Add regression tests for preprocessing boundaries, scheduler steps, freezing, GPU evaluation, and baseline metrics. `tests/test_early_stopping.py`, `tests/test_trial_budget.py`, and `tests/test_sird_rebuild.py` show the pattern: plain `assert`, stubbed research dependencies, runnable as `python tests/<name>.py`.
- Use reconstructed IRD RMSSE consistently for early stopping, Optuna selection, and final reporting; tune and evaluate with identical epoch and scheduler budgets.
- Select hyperparameters and epochs on validation data once. If refitting on train plus validation, use the selected fixed epoch count without validation monitoring.
- Normalize source loss so all sources together have a configurable total weight, then test no-source and target-specific source-selection ablations.
- Compare private-only, hard-shared, joint, pooled, and naive baselines with equal search budgets over multiple seeds and rolling forecast origins.
- Verify corrected GPU evaluation, baselines, and statistical testing in the rerun before publishing comparisons.

## Remaining Blockers

- `pipeline/eval.py` computes Friedman chi-square incorrectly and uses signed one-sided post-hoc p-values.
- Optuna/early stopping select scaled SIRD-rate MSSE, while final neural results report reconstructed IRD RMSSE; optimization and final evaluation also have different default epoch schedules.
- ARIMA and SIRD baselines return scalar metrics where comparison logs require per-IRD values; the SIRD baseline also lacks a reliable adapter from standard pipeline data to three IRD-count columns.
- Split boundaries assume zero forecast horizon while datasets use 14-day horizons, leaving the requested validation/test portions with far fewer valid forecast windows.
- ARIMA-SIRD cannot unpack the standard eight-field neural dataset correctly, and its exogenous path expects incompatible three-column inputs.

## Improvement Opportunities

Distinct from the blockers above: these are not correctness defects but accuracy and
cost improvements found while reading the model, training loop, and Optuna objective.
Several also remove failure modes, so they overlap with the rerun design.

### Quick wins — training and tuning cost

Applied:

- The joint dataloader is iterated lazily instead of being materialised as a list, so an epoch no longer holds every batch of every member at once (`model/train.py`).
- The tuning objective no longer calls `posttrain_save_model()`. With `save_state=False` it only produced captum attributions, four figures and a spreadsheet per target per trial, none of which the search reads; the final evaluation still generates them for the selected parameters.
- `TrialWrapper.suggest_int`/`suggest_float` return the constant when a range has equal bounds. The default strides and dilations are `(1, 1)`, so TPE was modelling four dimensions carrying no information. Existing studies cannot be resumed across this change, since the recorded distributions differ.
- The learning-rate range test runs once per trial rather than once per cluster. It costs `0.5 * min_epoch` extra epochs over every member and ran for each of the roughly nine clusters, while every cluster of a trial builds the same architecture from the same parameters. `eval()` and `make_objective()` take `find_lr_once=True` and thread an `lr_cache` down to `ClusterModel`; both default to it, so the search and the final fit still choose their learning rate the same way. Pass `find_lr_once=False` for a per-cluster search.

Outstanding, in rough order of payoff per unit of work:

- `model/modules/main.py:358-381` recomputes the future representation over the whole seed window at every decoder step and uses only the last position, and reallocates the seed with `cat` plus a slice each step. Currently masked because `use_representation_future` defaults to false, which forces `seed_length` to 1.
- `model/general.py:206-216` hardcodes `clip_grad_norm_(..., 1)` with the autoclip implementation commented out, leaving `grad_clip_percentile` threaded through the constructor with no effect. Restoring autoclip and deleting the parameter both change behaviour, so this needs a decision rather than a patch.
- Automatic mixed precision is deliberately left off (`model/general.py:1019`, `:1315`), and the `autocast` path in `model/train.py:36` is ready for it. These models are small enough to be kernel-launch bound rather than FLOP bound, so AMP is not expected to pay for itself here.
- Not a defect: `model/general.py:185-192` sizes each OneCycleLR cycle at `0.5 * min_epoch` while `max_epoch` defaults to 150, but `scheduler.py:56-75` restarts the cycle with `autodecay` shrinking `max_lr` by `sqrt(div_factor)`. The warm restarts are deliberate. What is untested is whether that cycle length suits the epoch budget.

### Quick wins — forecast accuracy

Applied:

- Reconstruction is bounded in `pipeline/sird.py::rebuild`, listed under Fixes Applied. Bounding the network's three outputs directly with `softplus` was rejected: the outputs live in the scaled space, and the scaler does not map raw zero to scaled zero, so a non-negative scaled output is not a non-negative rate. Clamping where the counts are built is both correct and shared by the SIRD baselines. A differentiable bound is still worth having for training, and is the IRD-space loss item under Big wins.
- `source_weights` now defaults to `(0.0, 1.0)` rather than `(0.5, 1.0)`, so the search can reach the low-transfer regime and produce a within-search no-transfer comparison.

Outstanding:

- Early stopping and the Optuna objective consume scaled SIRD-rate MSSE while results report reconstructed IRD RMSSE, so selection optimises a proxy of the reported metric through a nonlinear rebuild. Scoring the validation loader through the same `test()` path fixes it. This is the blocker above, listed here because it is also the cheapest accuracy gain. It is not a pure substitution: `EarlyStopping` compares the training loss against the validation loss, so both have to move to the same metric together.
- Teacher forcing is all-or-nothing per trial (`model/modules/main.py:393`, `:459-462`). Scheduled sampling, decaying the forcing probability across epochs, targets the exposure bias expected at a 14-step horizon.

### Big wins

Applied:

- Optuna pruning is enabled. The objective reports the running validation objective after each cluster and raises `TrialPruned` when the pruner rejects it, so a hopeless trial stops instead of training every remaining cluster. Clusters are visited in a fixed order, so a given step holds the same clusters in every trial. `main.create_study()` supplies the matching defaults: `MedianPruner(n_warmup_steps=1)`, which prunes from the second cluster onward, and `TPESampler(multivariate=True, group=True)` so the sampler models the conditional space (`lr` exists only when `onecycle` is off, the shared parameters only when `use_shared` is on) instead of treating every parameter as independent.
- `count_trials_done()` now counts pruned trials. It previously counted them as undone, which was harmless while nothing pruned but would have made `main.optimize()` loop until `n_trials` trials survived pruning. Covered by `tests/test_trial_budget.py`.

Outstanding:

- Batch the cluster members. `model/train.py:39-48` runs each member model sequentially, and each performs a 14-step autoregressive decode of small cells, so a training step is dominated by kernel-launch latency rather than compute. Member models are architecturally identical, so the member dimension can be folded into the batch with stacked per-member weights driven by `bmm` and grouped convolutions, with the shared branch run once over the concatenated batch. This is the largest available speedup. `torch.func.stack_module_state` with `vmap` would express it directly but needs a newer torch than the pinned 1.8.
- Pruning currently acts only at cluster boundaries, which is the coarsest useful granularity. Per-epoch reporting would prune sooner but early stopping gives each trial a different epoch count per cluster, so a global epoch step would compare different clusters across trials. Reporting per epoch within a fixed step budget per cluster would fix that.
- Replace the recursive decoder with a direct multi-horizon head emitting all 14 steps at once. This removes exposure bias and the sequential launch cost together, and direct strategies are usually competitive at this horizon.
- Compute the loss on reconstructed IRD counts rather than scaled rates, which requires a differentiable torch `rebuild`. A rate error is harmless at low infected counts and severe at high ones, so this aligns the optimisation target with the evaluation target.
- Rotate the cluster target instead of fixing it to the shortest training series (`pipeline/main.py:303-310`). Leave-one-city-out within each cluster multiplies evaluation data at unchanged per-fit cost and tests the transfer claim in both directions.
- Use the existing branch-freezing hooks. `freeze_shared()` and `freeze_private()` (`model/general.py:223-229`) are never called from either training loop, so source and target gradients compete over the shared weights every step. Fitting the shared branch on sources, freezing it, then fitting private branches per target is the standard multi-task schedule and is cheap to trial.
- Evaluate over rolling forecast origins and multiple seeds. This gates every item above: on a single split and seed, run-to-run variation exceeds the effects being measured.

### Literature-backed opportunities

Found by reading the architecture against the forecasting and multi-task literature. Items are
new unless marked as support for one already listed above. Nothing here is applied.

Each item is graded on whether its evidence comes from a setting resembling this one: roughly
38 daily series of a few hundred points, a 14-day horizon, models of a few thousand parameters,
and a single split. Most published gains do not, and several headline results have been overturned
by better-tuned baselines, so the grade matters more than the citation. Items whose evidence did
not survive that check are kept below under Weak or contested, not deleted, so the same papers are
not rediscovered later.

#### Quick wins — cost

- **Fuse the past encoder.** `model/modules/head.py:70-80` runs `nn.LSTMCell` in a Python loop over
  the whole past window (30 to 34 steps) for the private head and again for the shared head. Unlike
  the decoder, this loop is not autoregressive: the whole input sequence is known up front, so
  `nn.LSTM` over the packed sequence computes the same recurrence with cuDNN fusing the input gemms
  across timesteps. That removes roughly 30 of the 44 sequential cell calls per member per forward,
  in the regime the profile is dominated by (kernel-launch latency, not FLOPs). The learnable
  `hx_0`/`cx_0` become the `(h_0, c_0)` arguments, and `use_last_past=False` becomes a `[:-1]` slice
  of the input. Larger and far cheaper than the member-batching item under Big wins, and independent
  of it. *Evidence: not a research claim at all. It is an identity plus a documented engineering fact,
  and the speedup is measurable on this machine in an afternoon without any experiment design. The
  strongest item on this list, and the only one whose payoff does not depend on the rerun.*
  ([PyTorch RNN fusion](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/),
  [LSTM vs LSTMCell](https://discuss.pytorch.org/t/speed-of-lstm-vs-lstmcell/11868))
- **TorchScript the decoder cell.** If the decoder stays recursive, `torch.jit.script` on the cell body
  lets the fuser merge the pointwise gate group and cut the idle gap between the gemm and pointwise
  kernels, the same gap the PyTorch RNN work measured. Available in the pinned torch 1.8, unlike
  `torch.func`. CUDA graphs would capture the whole step but need a newer torch. *Evidence: same
  engineering source as above, but the reported figure is about 1.2x on an LSTM forward pass, so this
  is a modest gain behind a scripting constraint on the module, not a headline one.*
- **Consider Hyperband or successive halving, but do not expect much.** `main.py:140` uses `MedianPruner`,
  and Optuna's documentation reports median pruning as outperformed by SHA and Hyperband. *Evidence:
  weak here. Hyperband's advantage was established with hundreds to thousands of configurations and
  many fidelity rungs; this study prunes at roughly nine cluster boundaries and cannot afford more than
  dozens of trials, which is the regime where a median rule over a small trial history is about as good
  as a rank ladder. Worth a one-line swap, not worth a study.*
  ([Optuna HyperbandPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html),
  [implementation notes](https://tech.preferred.jp/en/blog/how-we-implement-hyperband-in-optuna/))

#### Quick wins — accuracy

- **Ensemble the seeds already required.** The rerun design already demands multiple seeds for a valid
  comparison. Taking the median across those seeds costs one extra line. *Evidence: the best on this
  list. The US COVID-19 Forecast Hub evaluations are prospective, cover dozens of teams over two years,
  and are on this exact target variable at a comparable horizon, and they find an equally weighted
  median ensemble at least as accurate as any individual model. The M4 and M5 retrospectives agree, and
  the mechanism, variance reduction over noisy-data neural forecasts, is the one that applies here,
  where run-to-run variation is already suspected to exceed the effect being measured. The caveat is
  that a Hub ensemble pools different model families; pooling seeds of one model reduces less variance
  than that, so expect a smaller gain than the Hub numbers suggest.*
  ([Forecast Hub evaluation](https://www.medrxiv.org/content/10.1101/2021.02.03.21250974.full.pdf),
  [M4 practitioner's view](https://www.sciencedirect.com/science/article/pii/S0169207019301189))
- **Add the two baselines the field expects.** `COVIDhub-baseline` (last observed value, intervals from
  historical weekly differences) and a single-linear-layer forecaster on the flattened past window
  (DLinear-style). *Evidence: strong, but note what it does and does not say. DLinear's headline claim,
  that linear beats transformers, is contested: it gave the linear model a 336-step lookback and the
  transformer baselines 96, and under matched tuning PatchTST recovers the lead. That dispute does not
  touch the use here. Nobody disputes that a one-layer linear model is a strong, nearly free baseline,
  and the same fair-comparison literature is explicit that under-tuned baselines manufacture apparent
  gains, which is the failure mode this project most needs to avoid. Both baselines are minutes of work
  and a joint-learning claim that does not clear them is not a claim.*
  ([DLinear](https://arxiv.org/abs/2205.13504),
  [PatchTST](https://arxiv.org/pdf/2211.14730),
  [no champions in LTSF](https://arxiv.org/html/2502.14045v1),
  [Forecast Hub dataset](https://www.nature.com/articles/s41597-022-01517-w))
- Support for items already listed: scheduled sampling with an inverse-sigmoid decay is the standard
  fix for the all-or-nothing teacher forcing; the direct multi-horizon head is the Direct/MIMO branch
  of the Ben Taieb and Bontempi strategy comparison. Their `rectify` hybrid, a recursive base model
  plus a direct correction per horizon, is a cheaper middle option than replacing the decoder outright.
  *Evidence: moderate. The strategy comparison is one competition, 111 daily series of cash withdrawals,
  and it reports direct and MIMO ahead of recursive by a margin that varies by series rather than a
  decisive win. The theoretical statement, that recursive is asymptotically biased and direct is not,
  is not in dispute, and the 14-step horizon is where that bias shows. Treat as a well-founded
  hypothesis to test in the rerun, not a known improvement.*
  ([strategy review](https://www.sciencedirect.com/science/article/abs/pii/S0957417412000528),
  [rectify](https://robjhyndman.com/papers/rectify.pdf))
- **RevIN on each window.** Per-window reversible instance normalisation, subtract and restore each
  input window's own mean and standard deviation with a learnable affine. *Evidence: real but probably
  redundant here. RevIN is well replicated and near-universally adopted, but its gains are measured on
  raw level series whose mean drifts between train and test, and this pipeline does not feed the network
  raw levels: it predicts SIRD rates, which are already ratios, after per-group scalers and a
  per-cluster rescale aligned to the target. Most of what RevIN removes has been removed twice already.
  Cheap enough to trial as an ablation on the rate inputs, but do not rank it above the items above it.*
  ([Kim et al., ICLR 2022](https://openreview.net/forum?id=cGDAkQo1C0p))

#### Big wins

- **One global model with a kabko embedding, as a comparison arm.** The current design fits one model per
  member and shares through a branch. Montero-Manso and Hyndman show a single global model fitted across a
  group is no more restrictive than per-series models and that its complexity stays constant as the group
  grows, which is also what won M4 and M5. Here that means one network conditioned on a learned per-kabko
  embedding instead of roughly 38 models: less memory, one training loop instead of a member loop, and
  every city's data behind every parameter. *Evidence: the theory is a proof and holds regardless of set
  size, but the empirical wins it is cited for come from collections of thousands of series, and the
  paper's own bound gets loose as the group shrinks. Thirty-eight series, split further into clusters, is
  at the small end of where cross-learning has been demonstrated. The cost argument is unconditional
  though, and it is a different method rather than a fix, so it belongs in the comparison as the
  pooled-with-identity arm the rerun design already calls for, not as an expected improvement.*
  ([Montero-Manso and Hyndman, IJF 2021](https://arxiv.org/abs/2008.00444))

#### Weak or contested — checked and downgraded

- **Gradient surgery (PCGrad, CAGrad, GradNorm) on the shared branch.** Sources and target descend the
  shared weights jointly every step, which looks like the textbook conflicting-gradient setting.
  *Evidence: this is the item the literature review killed. A NeurIPS 2022 study across language and
  vision tasks found these methods yield no improvement over plain scalarization, a tuned weighted sum
  of task losses, and that the effect size of ordinary hyperparameter choices is orders of magnitude
  larger than the multi-task-optimisation effect, so gains are usually an artifact of under-tuned
  baselines. This project already runs the control they recommend: `source_weight` is exactly that
  scalarization weight and Optuna already searches it over `(0.0, 1.0)`. Expect no gain, and treat any
  observed gain as a tuning artifact until it survives a matched search budget.*
  ([Kurin et al., NeurIPS 2022](https://arxiv.org/pdf/2209.11379),
  [Revisiting scalarization](https://proceedings.neurips.cc/paper_files/paper/2023/file/97c8a8eb0e5231d107d0da51b79e09cb-Paper-Conference.pdf),
  [PCGrad](https://arxiv.org/abs/2001.06782), [CAGrad](https://arxiv.org/pdf/2110.14048))
- **Physics-informed residual loss.** Adding an SIRD ODE-consistency residual to the objective would be a
  differentiable bound on the rates, regularising them where data is thin instead of only clamping after
  the fact in `pipeline/sird.py::rebuild`. *Evidence: thin for forecasting. The compartmental-plus-network
  line is real and active, but most of it estimates time-varying parameters and reconstructs an observed
  wave, which is an in-sample fitting task, and the forecasting papers that do exist compare against
  pure-data-driven or pure-model-driven variants of themselves rather than against the naive baseline the
  Forecast Hub work shows is hard to beat. Also note the structural mismatch: a residual penalty prices
  ODE violation, it does not prevent it, so it does not remove the need for the clamps. Keep the
  differentiable IRD-space loss already listed as the primary goal and treat the physics residual as an
  optional extra term, not as its justification.*
  ([PINN for compartmental models](https://pmc.ncbi.nlm.nih.gov/articles/PMC11407682/),
  [MP-PINN](https://arxiv.org/pdf/2411.06781),
  [physics-informed deep learning for infectious disease forecasting](https://arxiv.org/abs/2501.09298))
- **Learn the source weights.** `model/train.py:73` splits `source_weight` uniformly across sources, and
  the multi-source COVID transfer literature weights source domains by similarity or validation
  contribution instead. *Evidence: weakest sourcing on this list. Those papers are typically single-region
  case studies with hand-picked sources, no naive baseline, one split and one seed, which is the same
  evidentiary standard this investigation exists to reject. The idea is still the natural extension of
  the existing `SourcePick.CLOSEST` and costs little, but it should be justified by the ablation this
  project runs, not by those citations.*
  ([multi-source deep transfer](https://pmc.ncbi.nlm.nih.gov/articles/PMC9354391/))

#### Second review pass: efficient attention, curriculum, learned losses, resampling

A separate reading pass over efficient-architecture, curriculum-learning, loss-learning, active-learning
and class-imbalance work. Most of it does not transfer, and the reasons are recorded so the same
literature is not revisited. Three items survived.

Surviving:

- **Imbalanced-regression reweighting (LDS).** Epidemic windows are severely imbalanced in target
  magnitude: most windows are low-incidence and the rare peak windows are the only ones anyone cares
  about. Label distribution smoothing convolves a Gaussian kernel over the empirical target density and
  reweights the loss by the smoothed inverse density, which is a few lines and no architecture change.
  *Evidence: moderate, and the closest transfer available. The method is validated on single-value
  targets (age, depth, text similarity), so a 14-step vector target requires choosing what the density
  is over. It also partly overlaps machinery already present: MSSE divides by a naive-forecast
  denominator, which already absorbs some scale imbalance, so expect less than the published gain.*
  ([Yang et al., ICML 2021](https://arxiv.org/pdf/2102.09554))
- **Curriculum over the forecast horizon.** Grow `future_length` from 1 toward 14 across epochs. It
  attacks the same exposure bias as the scheduled-sampling item above, and unlike sample-level curricula
  it makes early epochs strictly cheaper, since a shorter decode is fewer sequential cell calls in the
  kernel-launch-bound regime that dominates here. *Evidence: this is the one curriculum variant reported
  as standard for time series rather than for text or images, but it is a training-schedule heuristic
  with no strong result behind it in this setting. Cheap to trial, and its cost saving is certain even
  if its accuracy effect is not.*
  ([data-centric time series review](https://arxiv.org/pdf/2404.16886))
- **Frequency-domain filtering, as accuracy rather than efficiency.** COVID reporting carries a hard
  weekly cycle (weekend reporting dips) that a 34-step convolution plus LSTM learns awkwardly and a
  learned spectral filter captures directly. The image-domain global filter idea continues into
  forecasting through FEDformer, GLFNet and FilterTS. *Evidence: speculative here. All of it is
  benchmarked on long-horizon multivariate datasets far larger than this one, and none of it is
  epidemic data. The argument for trying it is the inductive-bias match to a known reporting artifact
  in this specific data, not the published numbers.*
  ([GLFNet](https://dl.acm.org/doi/10.1145/3627673.3679579),
  [FilterTS](https://arxiv.org/html/2505.04158),
  [forecasters are frequency filters](https://arxiv.org/html/2411.01623))

Rejected, with the reason, so they are not reconsidered:

- **Efficient attention (Nyströmformer and relatives).** Linear-complexity attention pays off from
  roughly 512 tokens. The past window here is 30 to 34 and the architecture contains no attention at
  all. There is no quadratic term to remove.
- **Low-rank weight factorization (LoRaLin, EdgeFace).** Factorizing an `m x n` matrix at rank `r` only
  saves parameters when `r < mn/(m+n)`. The largest matrix here is the LSTM gate block at roughly
  `224 x 37` given a state size of at most 56 and hidden size of at most 37, so any saving needs
  `r < 32` on a model that is not memory-bound to begin with. EdgeFace targets a 1.77M-parameter face
  model on phone hardware; this model is orders of magnitude smaller.
- **Loss function learning (stochastic loss functions, discriminative adversarial losses, meta-learned
  and evolved losses).** Each adds an outer loop on top of an Optuna outer loop whose budget is already
  the binding constraint, so cost multiplies. More importantly the loss problem in this project is a
  known mismatch, selection on scaled SIRD-rate MSSE against reporting on reconstructed IRD RMSSE, and
  learning a loss to compensate for a mismatch that can be fixed by using the correct loss is the wrong
  order of work.
- **Active learning and learned acquisition functions.** Active learning buys down labeling cost.
  Labels here are free, since the future arrives on its own. The only transferable component is the
  loss-prediction module as a per-sample difficulty signal, and dynamic instance hardness supplies that
  as an exponential moving average of per-sample loss at no extra compute.
  ([DIH, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/62000dee5a05a6a71de3a6127a68778a-Abstract.html))
- **Sample-level curricula (competence-based, norm-based, spaced repetition, transfer-ranked).** The
  headline numbers are real, up to 70% less training time, 2.9x to 4.8x speedups, 34% to 50% of the data
  per epoch, but every one comes from a regime where data volume is the bottleneck: WMT translation
  corpora, ImageNet, CIFAR. Here an epoch over a few hundred points per city is already cheap and the
  cost is trials times epochs times members. Dropping samples per epoch on a dataset this small trades
  signal that cannot be spared for time that was not being spent.
  ([Platanios et al., NAACL 2019](https://arxiv.org/abs/1903.09848),
  [Liu et al., ACL 2020](https://arxiv.org/abs/2006.02014),
  [Amiri et al., EMNLP 2017](https://aclanthology.org/D17-1255/),
  [Hacohen and Weinshall, ICML 2019](https://proceedings.mlr.press/v97/hacohen19a.html))
- **Generative oversampling (SMOTE and its deep successors: conditional tabular GANs, VAE-based
  oversamplers, translation GANs).** All of it is classification on tabular data and does not transfer
  to a 14-step regression target. One of these papers argues against the family directly: deep
  generative oversampling beats SMOTE by margins that are significant in rank but "minor in absolute
  terms", and most of the gain came from undersampling the majority rather than from the generative
  model. The regression-side continuation of this line is the LDS item above.
  ([Camino et al., 2020](https://arxiv.org/abs/2005.03773))
- **Cognitive-science sources (flow theory, chunking, the zone of proximal development, hierarchical
  chunking models).** These motivate the curriculum family rather than specifying a method, and the
  chunking model is a representation learner rather than a forecaster. Useful as thesis framing, with
  nothing to implement.

#### On memory specifically

These models are tiny: hidden sizes 3 to 37, states 3 to 56, batches 16 to 512. Nothing here is
memory-bound, and no activation-memory technique (checkpointing, AMP, offload) is worth its complexity.
The one real VRAM consumer is that whole splits are materialised as CUDA tensors under the global default
tensor type, which is why the dataset-memoisation idea was dropped in Notes below. If VRAM ever binds,
that allocation, not the model, is the thing to move.

### Notes

- The default `n_trials` of 10000 in `main.optimize()` is not a reachable budget at current per-trial cost, even after the changes above.
- Not a defect, contrary to an earlier note here: `check_conv_kwargs` does not reject trials at the default ranges. With `stride` and `dilation` fixed at 1 it requires `kernel_size <= (past_length + 1) / 2`, and `past_length` is 30 to 34 against kernels of at most 14. The same holds for the future representation, which is disabled by default anyway.
- The objective now averages over the clusters it actually trained, so `debug` no longer divides by the full cluster count.
- Dropped from the cost list as not worth their churn: rebuilding the sliding windows with `sliding_window_view`, pre-stacking each split instead of using a `DataLoader`, and memoising `preprocessing_5`/`preprocessing_6` across trials. All three are seconds of window building against minutes of training per trial, the first two change the interface between `slice_dataset` and every `label_dataset_*`, and the third would hold `datasets_torch` for every parameter combination it has seen. Those tensors are allocated under the CUDA default tensor type, so the cache would compete with the model for VRAM.
- The applied training-loop, pruning, and learning-rate changes are syntax-checked only, apart from the three `tests/` scripts. No environment on the development machine has torch installed, so they have not been executed; run them in the pinned Python 3.8 experiment environment before trusting a rerun.
