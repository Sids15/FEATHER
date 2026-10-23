 INDICON 2026, Paper ID 5201: Response to Reviewers

**Title:** Toward Label-Free Detection of Harmful Data Drift in Deployed Classifiers

We thank both reviewers for their careful reading. The submitted version
validated FEATHER only on a closed-form synthetic testbed, and both reviewers
identified this as the main weakness. The revised manuscript therefore adds a
full evaluation on real datasets and trained deep networks. It also adds an
ablation study, significance tests, detection delay and computational cost,
an explicit statement of the assumptions, and an expanded limitations section.
Section numbers below refer to the revised manuscript.

 Reviewer 2

**R2.1: Include an ablation study analyzing the contribution of each component.**
Added Section VI-D, "Ablation Study", with Table III. It removes FEATHER's
components one at a time:
- **Fisher subspace replaced by a covariance subspace of the same dimension.**
  Blind drift is still detected, but the variant alarms on 28% of clean, 83%
  of benign and 89% of sensitive-control batches.
- **Subspace restriction removed (full-space mean shift).** The variant
  detects only 41% of blind-drift batches.
- **Held-out calibration removed.** FEATHER then alarms on every clean MNIST
  test batch; with held-out calibration the rate is 8%.

Only the full method both detects blind drift and stays silent on
output-visible drift.

**R2.2: Perform statistical significance testing.**
All results are means ± standard deviation over five independently trained
seeds. For the main comparison, benign false alarms of FEATHER against the
covariance ablation, Section VI-D now reports:
- an exact paired sign-flip test over seeds (p = 0.0625, the minimum possible
  with five seeds);
- an episode-level test: the gap is 34.4 points, with a 95% bootstrap CI of
  [27.9, 41.1], and a paired sign-flip test over the 62 benign episodes gives
  p < 10⁻⁴;
- the ablation alarms more often than FEATHER in 77% of benign episodes, and
  less often in none.

**R2.3: Expand the related work section.**
Section II is reorganized into five parts:
- concept drift detection;
- label-free performance monitoring;
- representation-space monitoring;
- Fisher geometry;
- an explicit research-gap paragraph.

It now includes recent work: PUDD (AAAI 2025), Komorniczak and Ksieniewicz
(2025), NEO (ICLR 2026), Amoukou et al. (NeurIPS 2024) and Zhang et al.
(2023). It also covers reliable ML systems (Sculley et al.). The conference's
15-reference limit applies.

**R2.4: Expand the methodology.**
- **Section IV-C, Computational Complexity (new):** the Fisher matrix factors
  through a C×C matrix, so both the offline fit and the online statistics
  scale linearly with the feature dimension.
- **Section V (new opening paragraph):** states the four assumptions of the
  method and justifies each one.
- **Algorithm 1:** now references the equations it implements.

**R2.5: Enhance the quality of the figures.**
- All figures are regenerated from the logged experiment data as vector PDF
  graphics, with Times New Roman labels in black at legible sizes.
- A new figure (Fig. 1) shows the measured Fisher eigenvalue spectrum.
- The conceptual schematics are replaced by plots of measured results.

 Reviewer 3

**R3.1: More extensive evaluation on real-world datasets and deployed models.**
The revision evaluates trained deep networks on four real datasets, five seeds
per model:
- **Rotated MNIST and CIFAR-10-C** (CNN and ResNet-18; 480 CIFAR-10-C episodes
  labelled by measured accuracy drop): Sections VI-B and VI-D.
- **Silent drift on the trained ResNet-18:** Section VI-C, Table I.
- **WILDS Camelyon17**, a real hospital-to-hospital shift on histopathology:
  Section VI-D.
- **A pre-registered CIFAR-100 subpopulation shift** (BREEDS protocol),
  reported honestly although its pre-registered criterion was not met:
  Section VI-D.

**R3.2: Clearer comparison with existing label-free techniques (accuracy,
false-alarm rate, cost, delay).**
FEATHER is compared with four output-based label-free monitors (confidence,
entropy, ATC and an error proxy), a covariance-subspace detector and the
full-space mean shift:
- **Detection accuracy:** AUROC is 1.0 for every method (Section VI-B), and
  every monitor detects 100% of harmful batches (Table II).
- **False-alarm rates:** clean, benign and gray alarm rates are in Table II.
- **Detection delay:** every monitor alarms on the first batch of harmful
  episodes. On gray episodes FEATHER alarms on the first batch, against
  1.17–1.47 batches on average (up to 10) for the output baselines
  (Section VI-D).
- **Cost:** FEATHER takes 3.0 ms per 500-image batch on a laptop CPU, against
  0.05 ms for the mean-shift gate (Section VI-D). The offline fit takes 11.1 s.

**R3.3: Explain and justify the assumptions behind the Fisher analysis and
the output-invisible subspace.**
- The new opening of Section V lists the assumptions: a frozen softmax head,
  a held-out reference set, exchangeable clean batches, and the spanning
  condition.
- The false-alarm bound for FEATHER's two-statistic rule is stated
  correctly as 2α. The submitted version gave it as α.
- Assumption 1 now carries its justification (it fails only on a set of
  measure zero) and is checked empirically (Fig. 1: exactly C−1 = 9 nonzero
  eigenvalues in all ten trained models).
- The empirical and true Fisher matrices share the same null space
  (Section IV-A).

**R3.4: Figures clear, high-resolution, readable, placed near first mention.**
All figures are vector graphics with legible labels and full captions. Each is
placed immediately after the paragraph that first refers to it.

**R3.5: Recent studies on concept drift, label-free monitoring and reliable
ML systems.**
See R2.3.

**R3.6: More comprehensive limitations, including mixed drift and
high-dimensional models.**
Section VII now covers:
- **mixed drift:** FEATHER responds only to the blind component, and the
  direction ratio reports its share, so FEATHER should run alongside an
  output monitor;
- **high-dimensional models:** cost grows linearly, and calibration needs
  more reference batches;
- **calibration mismatch;**
- **pure concept shift;**
- **the semi-synthetic nature of the silent-drift result.**

 Mandatory formatting instructions

- **Title:** in Title Case.
- **Authors:** no honorifics or designations; numeric affiliation
  superscripts.
- **Numbering:** all equations, figures and tables are numbered and referenced
  in the text.
- **References:** 15 references, all in a BibTeX file, cited with `\cite{}`,
  with no self-citations.
