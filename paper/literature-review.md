# Detecting Harmful Concept Drift Without Labels: A Literature Review

**Project:** FEATHER — Fisher-Eigenvalue Adaptive Thresholding for Error-Robust Drift Detection
*(Review-I literature review article — 2–3 pages when rendered.)*

## 1. Introduction

Machine learning models deployed in production face **concept drift**: the joint
distribution of inputs and labels changes after training, so a model that was
accurate at deployment silently degrades over time. Monitoring is complicated by
two facts of production life. First, **ground-truth labels arrive late or never**,
so accuracy cannot be tracked directly. Second, **not every distribution change
matters**: many shifts are *benign* (the model's accuracy is unaffected), while
some are *harmful* (accuracy collapses). A monitor that alarms on every
statistical change buries operations teams in false positives; one that waits for
labels reacts only after damage is done. This review surveys the three research
threads FEATHER builds on — statistical drift detection, label-free performance
estimation, and Fisher information geometry of neural networks — and identifies
the gap between them.

## 2. Statistical Drift Detection

The classical stream-mining literature treats drift detection as a statistical
hypothesis test on a data stream. **ADWIN** (Bifet & Gavaldà, 2007) maintains an
adaptive window and flags a change when two sub-windows differ significantly in
mean; **DDM** and its descendants monitor a classifier's *error rate* for
significant increases. A second family monitors distributions directly:
Kolmogorov–Smirnov windowing (KSWIN), Page–Hinkley CUSUM tests, and kernel
two-sample tests such as MMD.

Two limitations motivate the present project. First, error-rate detectors (DDM,
ADWIN as usually applied) **require labels** — exactly what production streams
lack. Second, distribution detectors are **performance-agnostic**: they test
whether *P(X)* changed, not whether the change matters, so they cannot separate
benign from harmful drift even in principle. Rabanser et al. (2019) demonstrated
this empirically in "Failing Loudly": univariate tests on learned representations
detect shift well, but detection strength correlates poorly with actual damage,
and the authors explicitly call out the open problem of characterizing *harmful*
shift.

## 3. Label-Free Performance Estimation and Harmful-Shift Detection

A more recent thread asks the sharper question directly: *how well is the model
doing right now, without labels?* Guillory et al. (2021) show that differences
of confidences (DoC) track accuracy across distribution shifts; Garg et al.
(2022) propose **ATC**, learning a confidence threshold on source data whose
exceedance rate estimates target accuracy. PUDD (AAAI 2025) uses prediction
uncertainty for early drift detection. Closest to our problem statement,
**Amoukou et al. (NeurIPS 2024)** formalize *sequential harmful shift detection
without labels*: a proxy model estimates the deployed model's errors, and
sequential testing controls false alarms. This work establishes the
harmful-vs-benign, label-free framing as a recognized research problem.

The common structural property of this entire family — DoC, ATC, uncertainty-
and proxy-based methods alike — is that their signal is a function of the
**model's outputs** (or of quantities trained to mimic them). This yields a
shared blind spot: *any drift that leaves the model's outputs unchanged is
invisible to them by construction.* Yet exactly such drift exists: whenever
input changes move a network's internal representations in directions the
classification head is insensitive to, softmax outputs — and hence confidence,
entropy, DoC, ATC, and error-proxies calibrated on outputs — do not move at
all, while the environment (and possibly the label function) has changed. No
method in this literature monitors that region.

## 4. Fisher Information Geometry of Neural Networks

The Fisher Information Matrix (FIM) quantifies how sensitively a model's
likelihood responds to perturbations. In deep learning it appears mainly as an
*optimization and compression* tool: Martens & Grosse (2015) introduced the
KFAC Kronecker factorization to make natural-gradient methods tractable, and
Fisher-based importance scores drive continual-learning regularizers and
pruning. Karakida et al. (2019) characterized FIM spectra in deep networks:
eigenvalues are dominated by a few large outliers while the bulk is near zero —
i.e., trained networks possess **large near-null Fisher subspaces**. Two recent
works bring Fisher-adjacent ideas to monitoring. Zhang et al. (Technometrics
2023) monitor the Fisher **score vector** (the gradient of the log-likelihood)
with multivariate control charts to detect drift in supervised models — a
detection signal, but not a harmfulness criterion, and a different mathematical
object from the FIM eigenstructure. Zero-Direction Probing (arXiv:2508.06776,
2025) uses null directions of activations to detect representational drift in
large language models — geometry related to ours, but theory-only, aimed at
LLM representation tracking, with no harmful/benign separation, no
task-supervised Fisher matrix, and no deployment-oriented evaluation.

## 5. The Research Gap

Placing the threads side by side exposes a precise gap:

| Thread | Monitors | Label-free? | Separates harmful/benign? | Sees output-invisible drift? |
|---|---|---|---|---|
| Statistical detectors (ADWIN/KSWIN/MMD) | inputs / error rate | partly | no | n/a (alarms on everything) |
| Output-based estimation (DoC, ATC, PUDD, Amoukou et al.) | model outputs / proxies | yes | yes (approximately) | **no — provably blind** |
| Fisher geometry (KFAC, Karakida, Zhang, ZDP) | curvature / scores | — | no | not applied to this problem |

No existing method uses the deployed model's own **task-Fisher eigenstructure**
to identify — exactly and in advance — the subspace of representation changes
its outputs cannot react to, and then monitors that subspace on the live
stream. That is FEATHER's contribution: the activation-space FIM of a softmax
head has rank at most the number of classes, so its complement (the *blind
subspace*) is large and computable exactly in closed form; drift confined to it
is provably undetectable by every output-based method above (a two-line
consequence of the geometry), while being cheap to monitor as a projection.
Combined with a classical statistical first stage (FEATHER-Lite cascade), this
yields a label-free monitor that covers precisely the region the state of the
art cannot see, with bootstrap-calibrated false-alarm control.

## 6. Conclusion

The literature has matured from "detect any shift" (Section 2) to "estimate
harm without labels" (Section 3), but the entire modern thread reads the
model's outputs and therefore shares one structural blind spot, while the
geometry needed to characterize that blind spot has existed all along in a
different community (Section 4). FEATHER connects the two: a
Fisher-geometric, provably-motivated monitor for silent drift, cascaded with
classical detectors for full coverage, and evaluated on controlled drift
benchmarks (Rotated MNIST, CIFAR-10-C) where harmfulness is measured, not
assumed.

## References

1. Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing. *SDM 2007*.
2. Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. *NeurIPS 2019*.
3. Guillory, D., Shankar, V., Ebrahimi, S., Darrell, T., & Schmidt, L. (2021). Predicting with Confidence on Unseen Distributions. *ICCV 2021*.
4. Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B., & Sedghi, H. (2022). Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. *ICLR 2022*.
5. Amoukou, S. I., Bewley, T., Mishra, S., Lecue, F., Magazzeni, D., & Veloso, M. (2024). Sequential Harmful Shift Detection Without Labels. *NeurIPS 2024*.
6. Early Concept Drift Detection via Prediction Uncertainty (PUDD). *AAAI 2025* (arXiv:2412.11158).
7. Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. *ICML 2015*.
8. Karakida, R., Akaho, S., & Amari, S. (2019). Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. *AISTATS 2019*.
9. Zhang et al. (2023). Concept Drift Monitoring and Diagnostics of Supervised Learning Models via Score Vectors. *Technometrics 65(2)*.
10. Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR 2019*.
11. Pandey, A. (2025). Zero-Direction Probing: A Linear-Algebraic Framework for Deep Analysis of Large-Language-Model Drift. *arXiv:2508.06776* (preprint).
