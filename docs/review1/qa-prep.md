# Review-I Q&A Preparation

Anticipated panel questions with correct, defensible answers. ⚠️ **Do not use
the Q&A answers from the old planning notes** — they contain the original
backwards intuition ("model has no information in thin directions, so it will
make errors there"), which a sharp panelist can dismantle. The corrected
geometry below is what our prototype actually verifies.

---

**Q1. "How do you know a drift is harmful without labels?"**
We don't claim certainty — we claim coverage of a provable blind spot. A
classifier's activation-space Fisher matrix splits its feature space into a
*sensitive* subspace (movement there changes outputs — existing monitors see
it) and a *blind* subspace (movement there changes outputs by exactly zero).
Drift into the blind subspace means the data is moving in ways the model
cannot react to: if the world changed there, the model fails *silently* — no
confidence drop, no output shift, no alarm from any existing label-free
monitor. FEATHER raises the flag precisely in that regime, and our experiments
measure (not assume) how often blind-subspace drift coincides with real
accuracy loss.

**Q2. "Isn't low Fisher information exactly where the model is *insensitive*?
Then how can movement there hurt it?"** *(the trap question — embrace it)*
Correct — that's our point, stated precisely. Movement in the blind subspace
doesn't change the model's *outputs*; it changes the *world* while the model's
outputs stay frozen. Harm doesn't come from the outputs moving; it comes from
the outputs *failing to move* when they should have (the label function
changed, or the feature extractor is off its training manifold). That's why
this drift is dangerous — it is invisible to every output-based monitor, which
we prove in two lines and verify to 10⁻¹⁰ in our prototype.

**Q3. "Why not just monitor softmax confidence or entropy?"**
Because confidence is a function of outputs, and blind-subspace drift provably
leaves outputs unchanged — confidence monitors flatline on exactly the drift
we target. We include confidence, entropy, BBSD, and ATC as baselines, and our
headline experiment constructs silent-drift streams where all of them stay
flat while FEATHER fires.

**Q4. "How is this different from PCA / Mahalanobis distance on activations?"**
PCA uses the *data covariance* — where activations spread. The Fisher uses the
*task geometry* — which directions the classification head reacts to. They can
disagree: a high-variance direction can be output-blind and vice versa. We run
the exact ablation (identical pipeline, covariance swapped for Fisher) to
quantify the difference. Also, our blind subspace has closed-form meaning
(null(W) for the head), which PCA directions lack.

**Q5. "Computing a Fisher Information Matrix for a deep network is huge — how
will you do it on your hardware?"**
We don't compute the parameter-space FIM at all. Our monitoring matrix is the
Fisher with respect to the *penultimate activation* — a 512×512 matrix for
ResNet-18, computed **exactly** from one pass over held-out data and
eigendecomposed in milliseconds. No KFAC approximation is needed for the core
method (we cite KFAC only for a multi-layer extension).

**Q6. "What if the null subspace changes over time (stale FIM)?"**
The blind subspace is a property of the frozen deployed model (essentially
null(W) of its head), so it is exactly as stale as the model itself — if the
model isn't retrained, its blind subspace doesn't move. On retraining, we
recompute offline in milliseconds. We state the fixed-reference-window
assumption explicitly, as is standard in this literature.

**Q7. "Isn't this already done? Fisher + drift exists."**
Three works are close, none overlap: Zhang et al. (Technometrics 2023) monitor
the Fisher *score vector* — a different object, detection-only, no
harmfulness notion. Zero-Direction Probing (2025) uses activation null spaces
for *LLM representational drift* — theory-only, no task-Fisher, no
harmful/benign separation. Amoukou et al. (NeurIPS 2024) do harmful-shift-
without-labels — but via an output-trained error proxy, which inherits the
output blind spot. FEATHER is the first to monitor the task-Fisher blind
subspace on a stream, and we cite and compare against all three.

**Q8. "What exactly will you demonstrate at the next review?"**
A live dashboard: a stream of images, the frozen model's true accuracy
(evaluation-only labels), the three FEATHER statistics, and alarms — showing
benign drift ignored, visible drift caught by stage 1, and silent drift caught
only by FEATHER. The Tier-1 core already runs today with 38 passing tests.

**Q9. "What's your fallback if the deep-net experiments don't support the
hypothesis?"**
The deliverable degrades gracefully: the detector-blindness proposition, the
calibrated monitor, the cascade, and the benchmark with measured harmfulness
labels are contributions independent of how strongly blind-drift correlates
with harm — a negative result there is itself a publishable finding about the
limits of geometric monitoring, and we report it honestly.

**Q10. "Why is your false-alarm control principled?"**
Thresholds are bootstrap quantiles of each statistic's distribution over clean
reference batches: choosing the 99th percentile targets a ~1% per-statistic
false-alarm rate on stationary data — measured at ≤ 8% for the union in our
tests, tunable by the quantile. No hand-set magic thresholds.
