# Natural subpopulation shift (BREEDS-style CIFAR-100) — pre-registered design

Fixed on 2026-10-18, before any model was trained or any data collected.
Scored mechanically by `src/experiments/breeds_analysis.py`.

## Why this experiment

ICDCN reviewers asked for evidence that drift the outputs cannot see can be
harmful on real data with real labels. A perturbation of a fixed input cannot
show this: if its outputs and its label are both unchanged, so is its
correctness (Reviewer 2's argument, which we accept). Harm with unchanged
output statistics needs the stream to bring in *different* inputs, with their
own true labels, that the frozen head maps to the same output distribution.
Subpopulation shift does exactly that, with every label real.

## Setup

- Model: CIFAR ResNet-18, 20-way, trained on CIFAR-100 superclasses using 4 of
  the 5 subclasses per superclass (held out: the subclass with the largest fine
  id in each superclass — a fixed rule). Same recipe as the CIFAR-10 model,
  5 seeds.
- Reference: seen-subclass train images (geometry/calibration 50/50 split).
- Episodes (8000 images each, batches of 500):
  - `clean`: seen-subclass test images;
  - `mix_RRR`: fraction rho in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0} of each
    batch from held-out subclasses (train + test images, never trained on);
  - `swap_<superclass>`: one superclass's images replaced by its held-out
    subclass, class mix unchanged (20 episodes).
- Monitors, all at the same 99th-percentile calibrated budget: FEATHER,
  covariance ablation, full-space mean shift, confidence, entropy, ATC, error
  proxy.

## Criteria (averaged over 5 seeds)

- **Primary.** Some rho has an accuracy drop >= 10 points while every output
  baseline alarms on < 20% of batches and FEATHER on > 80%.
- **Secondary.** For each monitor, the smallest rho with >= 50% alarms and the
  accuracy drop already incurred there.
- **Diagnostic.** Spearman correlation, across the 20 swap episodes, between
  accuracy drop and FEATHER's blind-shift score / blind fraction s, next to the
  output baselines' scores.

## Reporting commitment

Whatever the outcome, it is reported. If the outputs see the shift as well as
FEATHER does, it is reported as a second natural output-visible shift (like
WILDS Camelyon17), and the silent regime stays labelled semi-synthetic.

## Outcome (added 2026-10-18, after the 5-seed run)

The first scoring calibrated the output baselines on the training reference,
which the model has memorized; they then alarmed on 100% of *clean* test
batches, so that scoring was invalid. Rescored with the paper's standing
protocol (output baselines calibrated on the first half of the clean
deployment episode via `refit_baselines.py --baseline-calibration
clean-heldout`; clean false alarms 0%), FEATHER unchanged (training-reference
calibration, 7% clean false alarms):

- **Primary: not met.** No rho has the output baselines below 20% while the
  accuracy drop is >= 10 points.
- **Secondary.** First rho with >= 50% alarms: output baselines 0.2 (7.8-point
  drop), covariance ablation 0.2, FEATHER 0.3 (12.0 points), mean shift 0.8
  (30.7 points).
- **Diagnostic.** Spearman with accuracy drop over the 20 swaps: output scores
  0.85-0.87, FEATHER score 0.39, blind fraction s -0.86.

Per the reporting commitment, this is reported as a second natural
output-visible shift; the silent regime stays labelled semi-synthetic.
