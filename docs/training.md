# Training on the Workstation — Setup & Run Guide

Target machine: RTX 4500 Ada (24 GB) + Core Ultra 9 285/285K, 32 GB RAM.

## 1. What to transfer (zip vs. keep as-is)

| Item | How to transfer | Zipped or not? |
|---|---|---|
| **Code** | Zip the `FEATHER` folder **excluding** `.venv/`, `data/`, `outputs/`, `logs/` — or push to GitHub and clone (preferred: keeps git history and the commit trailer workflow) | zip for transport, extract on arrival |
| `cifar-10-python.tar.gz` | copy the file as-is | **keep zipped** — torchvision reads the tar.gz directly, never extract |
| CIFAR-10-C | copy the single `CIFAR-10-C.tar` (one big file copies faster than 20 loose .npy), then on the workstation: `tar -xf CIFAR-10-C.tar -C <data-root>` | **must be extracted** on the workstation — the code reads the `.npy` files, not the tar |
| `MNIST/raw/` (4 files) | copy the folder as-is | either works — torchvision reads both gzipped (`.gz`) and plain idx files |

Resulting layout on the workstation (any location, e.g. `E:\FEATHER-data`):

```
<data-root>/
├── cifar-10-python.tar.gz
├── CIFAR-10-C/            (20 .npy files after extracting the tar)
└── MNIST/raw/             (4 idx files)
```

## 2. Where to set the dataset path (the only path you ever change)

Pick one — no code edits needed:

- **Environment variable** (recommended, set once per shell):
  ```powershell
  $env:FEATHER_DATA_DIR = "E:\FEATHER-data"
  ```
  To make it permanent: `setx FEATHER_DATA_DIR "E:\FEATHER-data"` (new shells only).
- **Per-command flag**: add `--data-root E:\FEATHER-data` to any training command.

If the path is wrong, the scripts fail immediately with a message telling you
exactly what's missing and where it looked — nothing trains on bad data.

## 3. Environment setup (once)

```powershell
cd <repo>
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA build
pip install -e .
```

Then **run the test suite before any training** (it validates the trainer,
checkpoint/resume, models, and path handling on this machine):

```powershell
python -m pytest
```

All tests must pass. Also sanity-check the GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Training runs

```powershell
# MNIST base model (Tier 2) — under a minute
python src\experiments\train_mnist.py --run-name mnist_seed0 --seed 0

# CIFAR-10 ResNet-18 (Tier 3) — ~5-10 min, ~93-94% val acc
python src\experiments\train_cifar10.py --run-name cifar10_seed0 --seed 0

# Multi-seed benchmark (paper requires >= 5 seeds)
0..4 | ForEach-Object { python src\experiments\train_cifar10.py --run-name "cifar10_seed$_" --seed $_ }
```

## 5. Resuming an interrupted run

Reuse the **same `--run-name`** and add `--resume auto` (picks the latest
checkpoint of that run) or `--resume <path-to-checkpoint>`:

```powershell
python src\experiments\train_cifar10.py --run-name cifar10_seed0 --seed 0 --resume auto
```

Checkpoints carry model, optimizer, scheduler, AMP scaler, and all RNG states,
so the resumed run continues the exact trajectory. CIFAR checkpoints are saved
every 5 epochs by default (`--checkpoint-every`), MNIST every epoch; the last
3 are kept plus `best.pt` (best validation accuracy) always.

## 6. What each run produces (paper-grade provenance)

```
logs/<run>.log                    full DEBUG log: hardware, config, every N batches
                                  (loss/acc/lr/imgs-per-sec), epoch summaries,
                                  checkpoint and resume events
outputs/<run>/
├── config.json                   exact hyperparameters + architecture + param count
├── env.json                      GPU name/VRAM, torch/python versions, git commit
├── metrics.csv                   per-epoch train/val loss & acc, lr, epoch seconds
├── summary.json                  best/final accuracy, best epoch, total wall-clock
├── final_model.pt                frozen weights + meta (input to FEATHER phase)
└── checkpoints/
    ├── epoch_XXXX.pt             resumable checkpoints (last 3)
    └── best.pt                   best-validation checkpoint
```

These map directly onto the paper's needs: `metrics.csv` → training curves;
`summary.json` + `env.json` → the runtime/feasibility table (Proof F) and the
experimental-setup section; `config.json` + logged seeds → reproducibility
statement; `final_model.pt` → the frozen model every FEATHER experiment uses.

## 7. Monitoring benchmark (after training — the paper's raw results)

```powershell
# Tier 2 — Rotated MNIST (7 angle episodes)
python src\experiments\run_monitoring.py --model outputs\mnist_seed0\final_model.pt --mode rotated_mnist --out-name monitor_mnist_seed0

# Tier 3 — CIFAR-10-C (clean + 19 corruptions x 5 severities = 96 episodes)
python src\experiments\run_monitoring.py --model outputs\cifar10_seed0\final_model.pt --mode cifar10c --out-name monitor_cifar10_seed0
```

Each produces `outputs/<out-name>/episodes.csv` — one row per stream batch
with true accuracy (labels used for evaluation only), the output-based
baselines (mean confidence, entropy), and the FEATHER + PCA-ablation monitor
statistics — plus `fit.json` (subspace dims, thresholds, offline/online
timing for the paper's feasibility table) and a full log.

By default the clean reference data is split 50/50: geometry (Fisher/PCA
fit) and a held-out calibration split (μ_ref + bootstrap thresholds) — the
corrected protocol from the paper's calibration finding (Sect. 6.7). Confirm
`"calibration_mode": "heldout"` in `fit.json`. `--calibration-mode
same_split` reproduces the legacy (leaky) numbers only.

**Held-out calibration rerun (all 10 runs, then run `pytest` too):**

```powershell
foreach ($s in 0..4) {
  python src\experiments\run_monitoring.py --model outputs\mnist_seed$s\final_model.pt --mode rotated_mnist --out-name monitor_mnist_seed$s
  python src\experiments\run_monitoring.py --model outputs\cifar10_seed$s\final_model.pt --mode cifar10c --out-name monitor_cifar10_seed$s
}
python src\experiments\analyze_monitoring.py
```

## 8. What to bring back from the workstation

Copy the whole `outputs/` and `logs/` folders back (small: a few hundred MB).
`final_model.pt` feeds any rerun; `episodes.csv` + `fit.json` are all the
analysis and paper figures need — the analysis runs on the laptop, no GPU.
