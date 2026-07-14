# Review-I Presentation — Slide Content (21 slides)

Paste-ready content for PowerPoint. Rubric mapping: slides 3–10 target
**Parameter A** (problem, purpose, need, novelty, objectives — 5 marks);
slides 13–17 target **Parameter B** (feasibility & constraints — 5 marks).
Keep each slide sparse — headline + 3-5 bullets; say the rest aloud.

---

### Slide 1 — Title
- **FEATHER: Fisher-Eigenvalue Adaptive Thresholding for Error-Robust Drift Detection**
- Team members (Names, SAP IDs, Roll Nos.) • Guide name • Group number

### Slide 2 — Agenda
- Background → Problem → Novelty → Objectives → Scope → Tools → Feasibility → Timeline → Outcomes

### Slide 3 — Background: Concept Drift
- Deployed ML models face changing data distributions ("concept drift")
- **Benign drift:** data changes, accuracy unaffected
- **Harmful drift:** accuracy collapses — often *silently*
- Ground-truth labels arrive late or never in production streams

### Slide 4 — The Problem
- Statistical detectors (ADWIN, KSWIN, MMD) alarm on **any** change → false-alarm floods
- Error-rate detectors (DDM) **need labels** → unusable in real time
- Modern label-free monitors (confidence, entropy, ATC) read the **model's outputs**
- **Structural blind spot:** drift that doesn't move the outputs is invisible to all of them — by construction

### Slide 5 — Purpose & Need
- **Purpose:** a label-free monitor that watches the exact region output-based methods cannot see
- **Need:** silent model failure is the costliest MLOps failure mode — no alarm, wrong predictions
- Reduces false-alarm review cost *and* catches what current monitors provably miss

### Slide 6 — Key Insight (the one-picture slide)
- A classifier's **activation-space Fisher Information Matrix** splits its feature space into:
  - **Sensitive subspace** (rank ≤ #classes): movement here changes outputs → visible to existing monitors
  - **Blind subspace** (everything else — large): movement here changes outputs by **exactly zero**
- FEATHER projects each streaming batch's activation shift onto the blind subspace and alarms on calibrated excess
- *(Insert `docs/review1/figures/geometry.png` — generated from the prototype by `src/experiments/review1_figures.py`)*

### Slide 7 — Literature & Research Gap
- Rabanser'19 (NeurIPS): shift detection ≠ harm detection — open problem
- Amoukou'24 (NeurIPS): harmful-shift-without-labels framing, via output-proxy — inherits the blind spot
- Zhang'23 (Technometrics): Fisher *score* monitoring — different object, no harmfulness notion
- ZDP'25 (arXiv): null-space drift for LLMs — theory-only, no task-Fisher, no deployment
- **Gap:** nobody monitors the model's own Fisher blind subspace on a live stream

### Slide 8 — Novelty
- **First** use of the task-Fisher blind subspace as a drift harmfulness/silence filter
- **Provable motivation** (Detector-Blindness proposition): blind-subspace drift cannot be seen by *any* output-based monitor — two-line proof, already verified numerically in our prototype
- Exact closed-form computation (d×d matrix) — no approximation needed for the core method
- Novel cascade (FEATHER-Lite): statistical sensitivity + geometric specificity

### Slide 9 — Objectives
- (The 4 objectives from the Topic Approval Form, verbatim)

### Slide 10 — Proposed System
- **Offline (once):** train/freeze classifier → compute activation-space FIM exactly → eigendecompose → store blind basis → bootstrap-calibrate thresholds
- **Online (per batch, ms):** activations → shift Δμ → blind projection → statistics (direction sₜ, magnitude mₜ, energy vₜ) → alarm
- **FEATHER-Lite:** cheap detector (KSWIN via River) fires on any shift → FEATHER classifies visible vs. silent

### Slide 11 — Current Progress (differentiator!)
- Working prototype **already implemented and tested**: 38 passing unit/integration tests
- Verified analytically on controlled 2D streams:
  - Blind subspace recovered exactly (cosine ≈ 1.0 with theory)
  - Blind drift changes softmax outputs < 10⁻¹⁰ (Proposition 1 holds)
  - Monitor catches 100% of post-onset blind drift; calibrated false-alarm rate ≤ target
- Repository with production standards: venv, pinned deps, TDD, CI-ready
- *(Insert `docs/review1/figures/prototype_demo.png` — real output from the running prototype)*

### Slide 12 — Scope
- **In:** streaming image benchmarks (Rotated MNIST, CIFAR-10 → CIFAR-10-C), frozen-model monitoring, cascade, dashboard, multi-seed benchmark
- **Out:** online FIM recomputation (documented assumption), LLMs, model retraining/adaptation policies
- **Users:** MLOps engineers, data scientists, ML researchers

### Slide 13 — Tools & Technology
- Python 3.13 (venv) • PyTorch (model, activations) • NumPy/SciPy (exact FIM, eigh)
- River (statistical detectors: ADWIN, KSWIN, Page-Hinkley) • Streamlit (dashboard)
- Datasets: MNIST, CIFAR-10 (torchvision), CIFAR-10-C (Zenodo) — all free
- Hardware: workstation with **RTX 4500 Ada (24 GB VRAM)** + Intel Ultra 9

### Slide 14 — Technical Feasibility
- Core math is exact linear algebra on a d×d matrix (d = 512) — milliseconds, no approximation risk
- ResNet-18 on CIFAR-10 trains in minutes on our GPU; online monitoring is a matrix-vector product
- **Evidence: the Tier-1 core already runs and passes 38 tests** — feasibility is demonstrated, not estimated

### Slide 15 — Resource & Time Feasibility
- 100% free, open-source stack; datasets ≈ 3 GB total; zero API/cloud cost
- Compute headroom allows multi-seed statistics and full 19-corruption × 5-severity sweeps
- 5-month plan with the riskiest engineering (core math) already de-risked in month 0

### Slide 16 — Risk Assessment & Mitigation
- *Blind-drift ↔ harm correlation weaker than hypothesized on deep nets* → the benchmark itself + cascade remains a publishable contribution; harmfulness defined by measured accuracy, not assumption
- *Baseline (confidence/ATC) performs comparably on visible drift* → expected! Our claim is the **silent** drift regime, where they are provably blind — headline experiment isolates it
- *Theory over-runs* → main proposition already proven & verified; deeper bounds are stretch goals
- *Timeline slip* → dashboard is independent of experiments; each tier is a self-contained deliverable

### Slide 17 — Gantt Chart
- (Insert chart from `docs/review1/gantt-chart.md` — render the Mermaid or rebuild in Excel)

### Slide 18 — Expected Outcomes
- (The 4 outcomes from the Topic Approval Form, verbatim)

### Slide 19 — Evaluation Plan
- Harmfulness labeled by **measured accuracy drop** (benign < 2%, harmful > 10%)
- Metrics: harmful-drift AUROC, FPR on benign episodes, detection delay, runtime — ≥ 5 seeds
- Baselines: confidence, entropy, BBSD, ATC, PCA/Mahalanobis; label-using detectors shown as upper bound only

### Slide 20 — Conclusion
- Existing label-free monitors share one provable blind spot; FEATHER is built from the model's own geometry to watch exactly that spot
- Theory (proven), prototype (running), benchmark plan (defined), timeline (feasible)

### Slide 21 — References + Q&A
- 8 verified references from the Topic Approval Form • "Thank you — questions?"
