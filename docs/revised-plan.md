# FEATHER — Revised Plan (Independent Technical Analysis)

This document supersedes the method-level details in `docs/project-analysis.md`
(which faithfully summarizes the original AI-chat plan). Below: what breaks in the
original plan, the corrected formulation, the sharpened research thesis, and the
revised experiment/stack decisions. Capstone logistics, timeline, and Review-I
requirements are unchanged.

---

## 1. Problems Found in the Original Plan

### P1 — Space mismatch in the core formula (critical)
The notes define the FIM over **parameters** ( I = E[∇_θ log f · ∇_θ log fᵀ] ) but
then project **Δμ, an activation-space vector**, onto its eigenvectors. Those live
in different vector spaces with different dimensions — the projection as written is
undefined. Every downstream claim inherits this bug.

**Fix:** define the monitoring matrix directly in activation space — the Fisher
information **with respect to the penultimate activation φ** (treating φ as the
quantity the log-likelihood depends on):

    F_act = E_{x,y~D_ref} [ ∇_φ log p(y|φ(x)) · ∇_φ log p(y|φ(x))ᵀ ]  ∈ ℝ^{d×d}

For a softmax head z = Wφ + b: ∇_φ log p(y|φ) = Wᵀ(e_y − p). Now Δμ and the
eigenvectors live in the same d-dimensional space and the projection is well-defined.

### P2 — The exact structure of F_act changes the story (critical, and good news)
Since every gradient is of the form Wᵀv with v ∈ ℝ^C, **rank(F_act) ≤ C** (10 for
CIFAR-10). So for d = 512, the "null space" is ≥ 502-dimensional and is essentially
**null(W) — the subspace the classifier head is completely blind to**: Δφ ∈ null(W)
⇒ Δz = WΔφ = 0 ⇒ logits, predictions, confidences all *unchanged*.

The original intuition ("model has no information there → errors") is backwards
under first-order analysis — outputs are *insensitive* to null directions. The
correct, sharper story:

> Drift confined to the null space is **invisible to the model's outputs by
> construction**. No confidence-, entropy-, or prediction-based monitor (BBSD, ATC,
> PU-index on predictions) can ever detect it — yet the world has changed, so the
> frozen decision rule may now be wrong. FEATHER monitors exactly this blind spot.

### P3 — Theorem 2 as stated is likely unprovable
"Large null projection ⇒ accuracy has dropped by at least δ_acc" cannot follow from
first-order geometry (see P2 — null-space motion doesn't force errors; it *permits*
silent ones when labels shift too). Chasing this proof for 12 weeks is the plan's
biggest schedule risk. Replace with the theory package in §3.

### P4 — s_t normalization pathology
s_t = ‖proj_null Δμ‖/‖Δμ‖ is scale-invariant: a negligible noise-level Δμ pointing
into the null space yields s_t ≈ 1 → false alarm with no drift at all. The statistic
needs a magnitude gate and principled calibration (§4).

### P5 — Mean shift alone is too weak
A drift that preserves the mean but *spreads* into unseen directions (variance
change) is invisible to Δμ. Add a second-moment statistic (§4).

### P6 — Baseline category errors
ADWIN and DDM as used in the literature monitor the **error rate — they require
labels**. Benchmarking "label-free FEATHER" against label-using detectors is an
apples-to-oranges comparison reviewers will reject. Worse, the plan omits the
strongest label-free baselines: **confidence/entropy monitoring, BBSD (Rabanser
2019), ATC (Garg et al. 2022)**. "Why not just watch softmax confidence?" is a more
dangerous reviewer question than "why not PCA?" — and P2 gives us the principled
answer (null-space drift provably never moves confidence).

### P7 — Benign/harmful assignments are hand-waved
"Fog = benign, pixelate = harmful" is folklore; actual accuracy under CIFAR-10-C
varies by severity (fog at severity 5 hurts). Harmfulness must be **defined by
measured accuracy drop**, not by corruption name.

### P8 — Stack risks
scikit-multiflow is unmaintained (successor: **River**). NNGeometry/KFAC is
unnecessary for the core method once P1 is fixed: F_act is d×d (512×512), computable
**exactly** from one pass over held-out data and eigendecomposed in milliseconds.
Whole-network KFAC becomes an optional extension, not a dependency.

---

## 2. Corrected Method

**Offline (once, labeled held-out set):**
1. Train/freeze ResNet-18 on clean CIFAR-10; collect penultimate activations φ(x).
2. Compute exact F_act (d×d) and the activation covariance Σ_act (for the ablation).
3. Eigendecompose F_act → split ℝ^d into sensitive subspace S (top eigenvectors,
   captures ≥ (1−τ) of trace) and blind subspace N (the rest, incl. null(W)).
4. Calibrate thresholds by bootstrap: sample many clean reference batches, record
   the null statistics' empirical distribution, set thresholds at the q-th
   percentile (e.g., 99%) → direct, tunable false-alarm-rate control. (Replaces the
   vague "calibrated sigmoid".)

**Online (per batch, label-free, milliseconds):**
- Δμ_t = mean(φ(batch)) − μ_ref
- **s_t (direction):** ‖P_N Δμ_t‖ / ‖Δμ_t‖ — where is the drift pointing?
- **m_t (magnitude):** ‖P_N Δμ_t‖ in Mahalanobis units of the calibration
  distribution — is it big enough to matter? (fixes P4)
- **v_t (energy):** mean ‖P_N (φ(x) − μ_ref)‖² over the batch — catches
  variance/shape drift with no mean shift (fixes P5)
- Alarm when m_t (or v_t) exceeds its bootstrap threshold; s_t reported as the
  harmfulness-direction diagnostic.

**FEATHER-Lite cascade (unchanged in spirit, corrected in parts):**
Stage 1 = cheap **label-free** shift detector on a scalar summary stream (KSWIN via
River on projected activations, and/or BBSD on softmax outputs). Stage 2 = FEATHER
blind-subspace check to classify the shift as output-visible vs. silent and gate the
alarm. ADWIN/DDM appear only in a clearly-labeled "label-available upper bound" row.

---

## 3. Revised Theory Package (provable, still novel)

1. **Proposition 1 (Detector Blindness — new selling point).** If the activation
   shift is confined to null(W), the model's output distribution is *identical*,
   hence **any monitor computable from model outputs provably cannot detect the
   drift** (BBSD, confidence, entropy, ATC, prediction-distribution tests).
   Proof: two lines (Δz = 0). Positions FEATHER as *necessary*, not just better.
2. **Proposition 2 (Benign-direction guarantee — Theorem 1 reframed).** A shift of
   bounded magnitude confined to the sensitive subspace S changes expected
   log-likelihood by a bounded, calibratable amount (first-order expansion +
   smoothness of softmax cross-entropy). Manageable proof.
3. **Empirical hypothesis (replaces Theorem 2).** Blind-subspace drift energy
   correlates with silent accuracy degradation. Established by experiments, stated
   honestly as the empirical contribution. (Notes already sanctioned this fallback;
   we make it the plan, saving ~12 weeks of likely-doomed proof work.)
4. Theorem 3 (streaming eigen-tracking): dropped, cite streaming-SVD literature —
   unchanged from notes.

## 4. Revised Experiment Design

- **Ground-truth harmfulness by measurement (fixes P7):** for every (corruption,
  severity) stream, measure the frozen model's true accuracy; label episodes
  benign (drop < 2%), gray (2–10%), harmful (> 10%). Report AUROC for detecting
  harmful episodes, FPR on benign episodes, detection delay, all over ≥ 5 seeds.
- **Baseline suite (fixes P6):**
  - Label-free, output-based: max-softmax confidence, entropy, BBSD, ATC.
  - Label-free, feature-based: PCA/Mahalanobis low-variance projection (the
    Fisher-vs-PCA ablation stays — now well-posed: F_act vs Σ_act on identical
    pipelines), KSWIN/MMD on activations.
  - Label-using upper bound (separate table row): ADWIN/DDM on true error stream.
- **The headline experiment (new, from P2):** construct a *silent drift* stream —
  inject activation-space (or input-space, verified post-hoc) shifts confined to
  the blind subspace with label shift. Show every output-based baseline flatlines
  while FEATHER fires. This single figure is the paper.
- Tiers unchanged: 2D synthetic (exact math checks) → Rotated MNIST (drift
  continuum: small angles benign → large harmful) → CIFAR-10 + CIFAR-10-C (main).
- Runtime study (Proof F) unchanged. KFAC-validity study (old Proof G) replaced by
  a cheaper check: empirical Fisher vs. true Fisher on the 2D synthetic task.

## 5. Revised Stack

| Component | Original plan | Revised | Why |
|---|---|---|---|
| FIM | NNGeometry + KFAC (whole net) | **Exact d×d activation-space Fisher, plain PyTorch + NumPy/SciPy eigh** | P1/P8: correct space, no fragile dependency, milliseconds |
| Streaming/baselines | scikit-multiflow | **River** | scikit-multiflow abandoned |
| Model/data | PyTorch, torchvision, CIFAR-10-C | unchanged | fine |
| Dashboard | Streamlit | unchanged | fine |
| Extension only | — | NNGeometry/curvlinops KFAC for multi-layer FEATHER | ablation: does depth help? |

## 6. What This Buys Us

- A **well-posed method** (no space mismatch) that is *simpler* to implement.
- A **provable, two-line theorem** (Proposition 1) with genuine framing power,
  replacing a 12-week proof gamble.
- A **pre-built answer** to the two deadliest reviewer questions ("why not
  confidence?" — Prop 1; "why not PCA?" — F_act vs Σ_act ablation).
- **Fewer dependencies, less risk**, same demo, same capstone deliverables.

## 7. Novelty Positioning (merged from research findings)

`research/online-research-findings.md` (verified online, 2026-07-14) narrows the
claim but confirms the gap is open:

- **Amoukou et al., NeurIPS 2024** ("Sequential Harmful Shift Detection Without
  Labels") already owns the *problem framing* "label-free harmful vs. benign
  shift" — via a learned error-proxy + sequential testing. We must cite it, compare
  against it, and **never claim to be the first label-free harmful-drift detector**.
- **Zero-Direction Probing (arXiv 2025)** already uses null-space geometry for
  drift — but for LLM representational drift, theory-only, no harmful/benign
  separation, no vision experiments.
- **Zhang et al., Technometrics 2023** uses the Fisher *score vector* (different
  object) for drift detection — must cite to preempt "Fisher + drift exists."
- **PUDD/PU-index is real (AAAI 2025)** — usable as an alternative cascade stage 1.

**Defensible contribution statement:** *the first method to use the model's
task-Fisher blind subspace as a harmfulness filter for drift, with a provable
guarantee (Prop. 1) that the drifts it targets are invisible to all output-based
monitors — operationalized in a cheap cascade for streaming vision classifiers.*
FEATHER sits precisely in the gap between Amoukou et al. (framing, no geometry) and
ZDP (geometry, no harmfulness/deployment). Amoukou et al.'s error-proxy joins the
baseline suite (§4) if reimplementation cost allows.

All citations from the original notes were verified real (details + BibTeX in the
research file). Primary venue target: **ICLR 2026 "Catch, Adapt, and Operate"
workshop** (distribution-drift topic — direct fit); fallbacks: ECML-PKDD 2026,
DSAA 2026.

## 8. Unchanged

Review-I logistics and deliverables, the tiered dataset ladder (download links and
required layout: `docs/datasets.md` — user downloads manually, code never
auto-downloads), the cascade concept, the Streamlit demo, the semester timeline
(docs/project-analysis.md §9), rules.md conventions.
