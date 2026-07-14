# Topic Approval Form — Review-I

*(Fill team details — SAP IDs, Roll Nos., Names, Guide — from your registration form.)*

---

## 1. Project Title

**FEATHER: Fisher-Eigenvalue Adaptive Thresholding for Error-Robust Drift Detection**

## 2. Domain

Machine Learning / MLOps / Concept Drift Detection
*(Sub-domains: Information Geometry, Online Learning, Statistical Monitoring, Deep Learning)*

## 3. Project Objectives

1. To develop a **label-free drift monitoring framework (FEATHER)** that separates
   *output-visible* drift from *silent* drift by projecting streaming activation
   shifts onto the sensitive and blind eigen-subspaces of the deployed model's
   **activation-space Fisher Information Matrix**, computed exactly and once, offline.
2. To establish the framework's theoretical basis: a **Detector-Blindness
   proposition** proving that drift confined to the blind subspace cannot be
   detected by *any* monitor built on model outputs (confidence, entropy,
   prediction distributions), and a **benign-direction guarantee** bounding the
   log-likelihood change for shifts confined to the sensitive subspace.
3. To implement **FEATHER-Lite**, a cascaded detector combining a cheap
   statistical stage (KSWIN/ADWIN-class detectors via the River library) with
   FEATHER's geometric harmfulness filter, using **bootstrap-calibrated
   thresholds** for direct false-alarm-rate control.
4. To empirically validate on streaming benchmarks (synthetic 2D, Rotated MNIST,
   CIFAR-10 → CIFAR-10-C) that blind-subspace drift energy predicts *silent*
   accuracy degradation, benchmarked against label-free baselines (softmax
   confidence, entropy, BBSD, ATC, PCA/Mahalanobis) with a real-time Streamlit
   dashboard demonstration.

## 4. Motivation

Production ML models degrade silently when data drifts after deployment.
Existing statistical drift detectors flag *any* distribution change — including
harmless ones — producing costly false alarms, while error-rate detectors (DDM,
ADWIN-on-errors) require ground-truth labels that arrive late or never in real
streams. Recent label-free approaches estimate accuracy from **model outputs**
(confidence, entropy, agreement), but this creates a structural blind spot: any
drift that leaves the model's outputs unchanged is *invisible to them by
construction*, even when the world has changed underneath the model. There is
no existing method that monitors precisely this blind region of the model's
own geometry. FEATHER fills that gap: the model's activation-space Fisher
matrix identifies, exactly and cheaply, the subspace its outputs cannot see —
and watches it.

## 5. Expected Outcomes

1. An open-source, production-quality Python library (`feather`) — the offline
   Fisher/subspace computation and the online streaming monitor. *(A working,
   fully unit-tested prototype of the core already exists: 38 passing tests
   validating the geometry analytically on synthetic 2D streams.)*
2. A reproducible benchmark of harmful/silent-drift detection on Rotated MNIST
   and CIFAR-10-C, comparing FEATHER and FEATHER-Lite against confidence,
   entropy, BBSD, ATC, and PCA/Mahalanobis baselines across multiple seeds.
3. A live Streamlit dashboard visualizing the stream, true accuracy, FEATHER
   statistics, and alarms in real time.
4. A workshop/conference paper draft. Calendar-valid targets for this semester:
   **NeurIPS 2026 workshop track** (CFPs ~Aug–Sep 2026, stretch),
   **ICLR 2027 workshops** (~Jan–Feb 2027, primary), **ECML-PKDD 2027**
   (full paper, fallback).

## 6. Latest References (verified)

1. S. I. Amoukou, T. Bewley, S. Mishra, F. Lecue, D. Magazzeni, M. Veloso,
   "Sequential Harmful Shift Detection Without Labels," *NeurIPS*, 2024.
2. S. Rabanser, S. Günnemann, Z. C. Lipton, "Failing Loudly: An Empirical Study
   of Methods for Detecting Dataset Shift," *NeurIPS*, 2019.
3. S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, H. Sedghi, "Leveraging
   Unlabeled Data to Predict Out-of-Distribution Performance (ATC)," *ICLR*, 2022.
4. D. Hendrycks, T. Dietterich, "Benchmarking Neural Network Robustness to
   Common Corruptions and Perturbations," *ICLR*, 2019. (CIFAR-10-C)
5. Zhang et al., "Concept Drift Monitoring and Diagnostics of Supervised
   Learning Models via Score Vectors," *Technometrics* 65(2), 2023.
6. "Early Concept Drift Detection via Prediction Uncertainty (PUDD)," *AAAI*, 2025.
7. R. Karakida, S. Akaho, S. Amari, "Universal Statistics of Fisher Information
   in Deep Neural Networks: Mean Field Approach," *AISTATS*, 2019.
8. A. Bifet, R. Gavaldà, "Learning from Time-Changing Data with Adaptive
   Windowing," *SDM*, 2007. (ADWIN)

## 7. Status after Review 1

*(To be filled by the panel.)*
