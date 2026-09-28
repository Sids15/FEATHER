# FEATHER journal paper — design & plan

**Goal.** Turn the existing full manuscript (`paper/main.tex`) into a journal
submission for a **Springer journal** (Machine Learning / DMKD / Neural
Computing & Applications), typeset with the Springer Nature **`sn-jnl`** class,
extended with **two new experiments** (FEATHER-Lite cascade, WILDS natural
shift). I (Claude) write all experiment code; the workstation runs it; results
come back as CSV/JSON and I write the paper around them.

## Scope decisions (agreed 2026-09-28)
- Target: Springer journal, `sn-jnl` template (not SNmult, which is book/proceedings).
- Base: the existing `paper/main.tex` (already a full extension of the guarded
  conference versions — it holds the deep-net benchmark, the blind-drift proof,
  and the full theory that SCIS/COMSNETS deliberately omit).
- New experiments: **(A) FEATHER-Lite cascade**, **(B) WILDS Camelyon17**.
- Deferred (not now): intermediate-layer KFAC; extra baselines (Mahalanobis,
  BBSD); extra architectures. Add during revision only if a reviewer asks.

## ⚠ Timing / integrity (must resolve before journal submission)
The same core work is under review at SCIS 2026 (and earlier the deck listed
COMSNETS/ICDCN/INDICON). A journal will not review work concurrently under
review at a conference. The journal version must (a) go in only after the
conference version's status is resolved (accepted/withdrawn), (b) cite the
conference paper, and (c) carry significant new material — which cascade + WILDS
provide on top of the already-extended manuscript.

---

# Phase 1 — Experiment code (I write, you run)

Both reuse the existing, dataset-agnostic monitoring core in
`feather.monitoring` (`fit_monitors`, `run_episode`, `split_reference_dataset`,
`load_frozen_model`, `head_params`) and `feather.core.monitor.SubspaceDriftMonitor`.
Baselines (confidence/entropy/ATC/proxy/PCA) already exist — reused, not rewritten.

## A. FEATHER-Lite cascade  *(high feasibility; no new data/training)*
**New files:** `src/feather/cascade.py`, `src/experiments/cascade.py`, `tests/test_cascade.py`.

- **Stage 1 (cheap, always on):** a River detector (ADWIN or KSWIN; River is
  already a dependency) watching **mean softmax confidence** per batch. Fires on
  any change.
- **Stage 2 (FEATHER, only when stage-1 fires):** the existing
  `SubspaceDriftMonitor` sorts the flagged shift into *output-visible* (defer to
  the confidence monitor) vs *blind* (raise the harmful alarm).
- **Streams:** built from the existing CIFAR-10-C / Rotated-MNIST loaders as
  **clean→drift** episodes (K clean batches, then a corruption/rotation), so
  detection **delay** is measurable. Reuses trained models you already have; can
  reuse saved `raw/` activations to avoid re-inference.
- **Metrics → `outputs/cascade_<suite>/summary.json` + `episodes.csv`:**
  detection delay (batches from onset to stage-1 fire), false-alarm rate on
  clean & benign episodes, and **mean fraction of batches FEATHER runs** (the
  cost win) — all vs FEATHER-always and vs a stage-1-only detector.
- **You run:** one command per trained model; minutes, GPU optional.
- **Self-check (`test_cascade.py`):** on a synthetic clean→blind-drift stream,
  assert stage-1 fires after onset, stage-2 labels it *blind*, and cascade cost
  < FEATHER-always.

## B. WILDS Camelyon17  *(feasible; the bigger lift — needs data + training)*
**New files:** `src/feather/data/wilds.py`, `src/experiments/train_wilds.py`
(adapted from `train_cifar10.py`), a `--mode wilds_camelyon17` branch in
`src/experiments/run_monitoring.py`, `tests/test_wilds_data.py`.

- **Data:** the `wilds` package → Camelyon17 (~10 GB). Reference = in-domain
  train/val (train hospitals); **episodes = OOD test hospital(s)**, streamed in
  batches. Harm graded by **measured accuracy drop** on held-out labels (same
  benign/gray/harmful thresholds as the corruption suites).
- **Model:** ResNet-18 (or -50) trained on Camelyon17 train, saved as
  `final_model.pt` in the existing format (so `load_frozen_model` + `head_params`
  work unchanged). Binary task ⇒ rank(F) ≤ C−1 = 1 ⇒ the blind subspace is
  almost the entire feature space — a clean structural demonstration.
- **Output:** identical `episodes.csv` + a `monitoring_summary.json` entry under
  a new suite key `wilds_camelyon17` (AUROC + alarm rates per method/grade), so
  it drops straight into the existing result tables/figures.
- **You run:** `pip install wilds`; download Camelyon17; `train_wilds.py`
  (a few hours on the 24 GB GPU); then `run_monitoring.py --mode wilds_camelyon17`.
- **Honest framing:** natural WILDS shift is mostly *output-visible*, so this
  validates FEATHER's tracks-harm + false-alarm discipline on **real** data
  (what the Limitations asks for) — it is not the pure silent regime; the paper
  will say so.
- **Self-check (`test_wilds_data.py`):** loader returns correctly-shaped
  tensors, in-domain vs OOD splits are disjoint, labels in {0,1}. (Skips
  cleanly if the dataset isn't downloaded.)

## What you hand back
Just the `outputs/cascade_*/…` and `outputs/monitor_wilds_*/…` files (same
schema as `outputs/monitor_cifar10_seed0/`). No reshaping needed on my end.

---

# Phase 2 — The journal paper (`paper-journal/`, `sn-jnl`)

New folder `paper-journal/` (leave `paper/` and `paper-conference/` untouched):
port `paper/main.tex` to `sn-jnl.cls`, then complete and extend.

**Outline** (⋆ = new/expanded vs the current manuscript):
1. Introduction (expand: production framing, contributions incl. cascade + WILDS)
2. Related Work (expand: feature-space/OOD, label-free perf. estimation, Fisher-in-NN, drift benchmarks)
3. Problem Setup
4. The FEATHER Monitor (method + full pipeline figure)
5. Theory — rank bound, blindness (Prop.), false-alarm control; full proofs in-text/appendix
6. Experiments
   - Setup
   - Linear testbed (exact answers)
   - Rank structure appears exactly
   - Blind-subspace shift tracks harm
   - Benchmark saturation (AUROC ceiling)
   - Deep-feature silent drift (blind-drift proof)
   - ⋆ **Natural shift: WILDS Camelyon17**
   - ⋆ **FEATHER-Lite cascade** (delay / false alarms / cost)
   - Where the false alarms land (+ the covariance ablation)
   - A calibration finding
   - Cost
7. Discussion
8. Limitations (shrinks: cascade + WILDS move from "future" to "done")
9. Conclusion
- ⋆ Reproducibility statement (code, seeds, scripts, tests)
- ⋆ Appendix: full proofs

**Author block / metadata:** reuse the finalized SCIS author block
(Ahuja · Bandi · Sharma, STME/NMIMS Indore). Cite the conference version.

**Figures/tables:** the current manuscript has only 2 figures + 2 tables;
journal version adds figures from data already computed (Fisher spectra across
seeds, tracks-harm scatter, false-alarm bars) plus the two new-experiment
figures. All figures script-generated (no hand-drawn data).

## Work order
1. Write cascade code + test → you run it.
2. Write WILDS code + test → you set up data, train, run it.
3. In parallel with your runs, I port `paper/main.tex` → `paper-journal/` (sn-jnl)
   and draft the non-results sections (Intro, Related Work, Method, Theory).
4. As results arrive, write the two new experiment subsections + refresh tables/figures.
5. Polish, reproducibility, appendix; final compile check.
