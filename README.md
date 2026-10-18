# FEATHER

**Fisher-Eigenvalue Adaptive Thresholding for Error-Robust drift detection** — a
label-free monitor for deployed classifiers that watches the part of the
representation the model's outputs provably cannot see.

For a `C`-class softmax head, the Fisher information of the log-likelihood with
respect to the penultimate activation has rank at most `C − 1`. Its orthogonal
complement, the *blind subspace*, is the set of activation directions that move
every logit by the same amount, which the softmax cancels. Drift along those
directions changes no confidence, entropy, or error-proxy statistic, so no
output-based monitor can detect it. FEATHER computes that subspace exactly in
one offline pass and watches it online for the cost of one projection per batch,
with thresholds calibrated to a chosen false-alarm budget.

Output-invisible is not the same as harmful: FEATHER flags drift the outputs
cannot see, and whether that drift costs accuracy is **measured** from held-out
labels in every experiment, never assumed.

> Capstone project (MPSTME, NMIMS, 2026–27). Team: Rishabh Ahuja, Siddhant
> Bandi, Shruti Sharma.

---

## Repository layout

```
src/feather/            the library
  core/fisher.py          activation-space Fisher matrix and sensitive/blind split
  core/monitor.py         streaming monitor: direction ratio, magnitude, energy
  baselines.py            output baselines: confidence, entropy, ATC, error proxy
  cascade.py              two-stage cascade (cheap mean-shift gate -> FEATHER)
  monitoring.py           fit all monitors on a frozen model, stream episodes
  data/                   MNIST, CIFAR-10(-C), CIFAR-100 BREEDS, WILDS, 2-D toy
  models/, training/      CNN / ResNet-18 and a resumable trainer
src/experiments/        one script per paper result (see "Reproducing the paper")
tests/                  unit + integration tests (math core has analytic checks)
paper/                  journal manuscript (Springer SNmult)
  main.tex                12-page version
  main_full_22pp.tex      full manuscript
paper-conference/       conference versions; ACM-Journal/ is the full paper in acmart
docs/                   datasets, training guide, experiment designs
rules.md                project rules (environment, code quality, git)
```

## Setup

Python 3.10+ in a virtual environment (see `rules.md` §1).

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU build
pip install -e .
$env:PYTHONPATH = "src"       # lets analysis scripts import each other
python -m pytest              # must pass before any training
```

On a CPU-only machine use the `whl/cpu` index. The NumPy core, the analysis
scripts, and most tests run without torch; torch-dependent tests skip cleanly.

**Datasets are never downloaded by code.** Download them manually into one folder
and point `FEATHER_DATA_DIR` at it — layout and links in
[`docs/datasets.md`](docs/datasets.md):

```powershell
$env:FEATHER_DATA_DIR = "D:\FEATHER-data"
```

## Using FEATHER on your own model

All FEATHER needs is penultimate activations `phi`, labels `y` for a labeled
reference set, and the final layer's `weight` (`C × d`) and `bias`:

```python
from feather.core.fisher import activation_fisher, fisher_subspaces
from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

fisher = activation_fisher(phi_geometry, y_geometry, weight, bias)   # exact, d x d
blind = fisher_subspaces(fisher).blind_basis                         # d x (d - C + 1)
monitor = SubspaceDriftMonitor(blind, phi_calibration, MonitorConfig(batch_size=500))

result = monitor.score(phi_batch)   # per unlabeled stream batch
result.alarm, result.direction_ratio, result.shift_magnitude, result.energy
```

Fit the Fisher basis and calibrate the thresholds on **disjoint** halves of the
reference data (split the arrays; for torch datasets use
`feather.monitoring.split_reference_dataset`); calibrating on the fitting split
makes the thresholds optimistic.

## Reproducing the paper

Training and monitoring need a GPU; everything after that runs on a laptop from
the saved `outputs/`. Full workstation guide: [`docs/training.md`](docs/training.md).

**1. Train (5 seeds each).**

```powershell
0..4 | ForEach-Object { python src\experiments\train_mnist.py   --run-name "mnist_seed$_"   --seed $_ }
0..4 | ForEach-Object { python src\experiments\train_cifar10.py --run-name "cifar10_seed$_" --seed $_ }
0..4 | ForEach-Object { python src\experiments\train_wilds.py   --run-name "wilds_seed$_"   --seed $_ }
```

**2. Monitor** (writes `outputs/<out-name>/episodes.csv`, `fit.json`, `raw/`).

```powershell
foreach ($s in 0..4) {
  python src\experiments\run_monitoring.py --model outputs\mnist_seed$s\final_model.pt   --mode rotated_mnist    --out-name monitor_mnist_seed$s
  python src\experiments\run_monitoring.py --model outputs\cifar10_seed$s\final_model.pt --mode cifar10c         --out-name monitor_cifar10_seed$s
  python src\experiments\run_monitoring.py --model outputs\wilds_seed$s\final_model.pt   --mode wilds_camelyon17 --out-name monitor_wilds_seed$s
  python src\experiments\cascade.py        --model outputs\cifar10_seed$s\final_model.pt --mode cifar10c         --out-name cascade_cifar10_seed$s
}
```

**3. Analyse (no GPU).** Each paper result maps to one script:

| Paper result | Script |
|---|---|
| Harm tracking (Fig. 1), false-alarm table, AUROC | `analyze_monitoring.py` (after `refit_baselines.py --baseline-calibration clean-heldout`) |
| Mean-shift baseline row | `meanshift_baseline.py outputs\monitor_cifar10_seed*` |
| Deep-feature silent drift (Table 1, Fig. 2) | `blind_subspace_deep.py` |
| Matched (clean-test) calibration | `symmetric_feather.py` (needs a `--save-all-phi` monitoring run) |
| Fisher rank structure (9 of 512) | `spectrum_figure.py` |
| Run-rule (persistence) check | `persistence_rule.py outputs\monitor_cifar10_seed*` |
| Cascade cost and alarms | `cascade.py` summaries |
| Per-batch monitor cost (3.0 ms) | `monitor_cost.py outputs\monitor_cifar10_seed0` |
| 2-D testbed geometry and accuracy | `tests/test_integration_2d.py` |

Scripts that take run folders expect them listed explicitly; in PowerShell pass
`(Get-ChildItem outputs -Directory -Filter monitor_cifar10_seed*).FullName`.

## Natural subpopulation shift (BREEDS)

The open question from review: can drift the outputs cannot see be harmful on
**real data with real labels**? Perturbing a fixed input cannot show it (if its
outputs and label are unchanged, so is its correctness), so this experiment
brings in *different* inputs instead: a ResNet-18 trained on CIFAR-100's 20
superclasses using 4 of 5 subclasses each, deployed on the held-out subclass.
The design and success criteria were committed before any data was collected:
[`docs/breeds-design.md`](docs/breeds-design.md).

```powershell
# once: extract cifar-100-python.tar.gz into $env:FEATHER_DATA_DIR (docs/datasets.md §1b)
foreach ($s in 0..4) {
  python src\experiments\train_cifar10.py --dataset breeds_cifar100 --run-name breeds_seed$s --seed $s
  python src\experiments\run_monitoring.py --model outputs\breeds_seed$s\final_model.pt `
      --mode breeds_cifar100 --out-name monitor_breeds_seed$s
}
$runs = (Get-ChildItem outputs -Directory -Filter monitor_breeds_seed*).FullName
python src\experiments\refit_baselines.py --baseline-calibration clean-heldout --runs $runs
python src\experiments\breeds_analysis.py $runs
```

About an hour of GPU time on an RTX-class card. The refit calibrates the output
baselines on held-out clean deployment data, as everywhere else in the paper
(the analysis refuses runs without it). The last command prints the per-`rho`
alarm table and the primary / secondary / diagnostic verdicts, and writes
`outputs/breeds_summary.json`.

**Result (5 seeds):** the pre-registered primary criterion was not met. At
matched calibration the output baselines reach 50% alarms at rho = 0.2
(7.8-point drop), FEATHER at rho = 0.3 (12.0 points). Harmful subclasses shift
mainly along the *sensitive* directions (blind fraction vs. drop: Spearman
-0.86), so this natural shift is output-visible, like Camelyon17, and the paper
reports it as such.

## Results at a glance

ResNet-18 on CIFAR-10, mean over five seeds, every monitor at the same 1%
calibrated budget:

- **Exact rank structure:** 9 informative directions out of 512; the spectrum
  drops 14 orders of magnitude after the 9th eigenvalue.
- **Silent drift:** a blind-subspace shift changes no softmax output by more
  than 2×10⁻¹⁰ yet cuts a synthetic-rule accuracy from 99% to 7%. FEATHER alarms
  on 100% of batches, output baselines on none, and a plain mean shift on 41%.
- **False alarms on benign corruptions:** FEATHER 48%, covariance ablation 83%,
  mean shift 20%, output baselines 6–9%. On corruption benchmarks the drift is
  output-visible, so output monitors lead there; FEATHER's advantage is the
  regime they cannot see.
- **Real shift (WILDS Camelyon17):** 511 of 512 dimensions blind; FEATHER tracks
  the 99% → 58% hospital-to-hospital accuracy drop and holds its budget in
  distribution.

## Development

- Run `python -m pytest` before every commit that touches `src/`.
- Every commit ends with `Co-authored-by: rishabhahuja12 <rishabhahuja961@gmail.com>`
  (`rules.md` §5).
- Every number in the paper must trace to a script in `src/experiments/`.
