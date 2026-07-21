"""Bridge between trained PyTorch models and the NumPy FEATHER core.

Offline: load a frozen model (``outputs/<run>/final_model.pt``), extract
penultimate activations and the softmax head's (W, b), compute the
activation-space Fisher, and calibrate two monitors on clean reference data:

- **FEATHER**: blind subspace of the activation-space Fisher matrix.
- **PCA ablation**: identical pipeline with the Fisher matrix replaced by the
  activation covariance (lowest-variance subspace of the same dimension) —
  the paper's Fisher-vs-PCA control.

The reference data plays two distinct roles, served by two *disjoint* splits
(see :func:`split_reference_dataset`):

- **geometry**: fits the Fisher/PCA bases;
- **calibration**: supplies μ_ref and the bootstrap thresholds.

Calibrating on the geometry split leaks the fit into the thresholds and makes
them too optimistic (paper Sect. 6.7); passing the same dataset for both is
kept only to reproduce the legacy protocol.

Online: per streaming batch, record true accuracy (labels used for
*evaluation only*), the output-based baseline signals (mean confidence, mean
entropy), and both monitors' statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from feather.core.fisher import activation_fisher, fisher_subspaces
from feather.core.monitor import MonitorConfig, MonitorResult, SubspaceDriftMonitor
from feather.models import MODELS

logger = logging.getLogger("feather.monitoring")


def load_frozen_model(path: str | Path, device: torch.device) -> nn.Module:
    """Rebuild a model from a ``final_model.pt`` produced by the Trainer."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    meta = payload.get("model_meta")
    if not meta or meta.get("arch") not in MODELS:
        raise ValueError(
            f"cannot rebuild model from {path}: unknown architecture meta {meta!r}"
        )
    kwargs = {k: v for k, v in meta.items() if k != "arch"}
    model = MODELS[meta["arch"]](**kwargs)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info("loaded frozen model %s from %s", meta, path)
    return model


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, logits, labels) for a whole dataset as NumPy arrays."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, logits, labels = [], [], []
    for inputs, targets in loader:
        phi = model.features(inputs.to(device))
        features.append(phi.cpu().numpy())
        logits.append(model.head(phi).cpu().numpy())
        labels.append(targets.numpy())
    return np.vstack(features), np.vstack(logits), np.concatenate(labels)


def split_reference_dataset(
    dataset: Dataset,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Split reference data into disjoint (geometry, calibration) subsets.

    Deterministic given ``seed``: a fixed permutation of the indices, with the
    last ``calibration_fraction`` share going to calibration.

    Args:
        dataset: Clean reference dataset (indexable, with ``len``).
        calibration_fraction: Share of samples reserved for threshold
            calibration, in (0, 1).
        seed: Seed for the index permutation.

    Returns:
        (geometry, calibration) subsets; both non-empty.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    n = len(dataset)
    n_calibration = int(round(n * calibration_fraction))
    if n_calibration == 0 or n_calibration == n:
        raise ValueError(
            f"split of {n} samples at fraction {calibration_fraction} leaves "
            "an empty subset"
        )
    permutation = np.random.default_rng(seed).permutation(n)
    geometry_idx = permutation[: n - n_calibration].tolist()
    calibration_idx = permutation[n - n_calibration :].tolist()
    return Subset(dataset, geometry_idx), Subset(dataset, calibration_idx)


def head_params(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Return the softmax head's (W, b) as NumPy arrays."""
    head = model.head
    return head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class MonitorBundle:
    """Everything the online phase needs, fitted offline on reference data."""

    feather: SubspaceDriftMonitor
    pca: SubspaceDriftMonitor
    blind_dim: int
    fisher_eigenvalues: np.ndarray
    geometry_n: int
    calibration_n: int


def fit_monitors(
    model: nn.Module,
    geometry_reference: Dataset,
    calibration_reference: Dataset,
    device: torch.device,
    batch_size: int = 500,
    quantile: float = 0.99,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> MonitorBundle:
    """Fit the FEATHER monitor and its PCA-ablation twin on clean reference data.

    The Fisher/PCA bases come from ``geometry_reference``; μ_ref and the
    bootstrap thresholds come from ``calibration_reference``. The two must be
    disjoint for honest thresholds (use :func:`split_reference_dataset`);
    passing the same dataset twice reproduces the legacy same-split protocol.

    The PCA monitor uses the lowest-variance subspace of the activation
    covariance with the *same dimension* as FEATHER's blind subspace, so the
    two differ only in the matrix that defines the geometry.
    """
    phi, _, labels = extract_activations(model, geometry_reference, device)
    weight, bias = head_params(model)

    fisher = activation_fisher(phi, labels, weight, bias)
    subspaces = fisher_subspaces(fisher)
    blind_dim = subspaces.blind_basis.shape[1]
    logger.info(
        "activation Fisher: d=%d, sensitive=%d, blind=%d, top eigenvalue=%.4g",
        phi.shape[1], subspaces.sensitive_basis.shape[1], blind_dim,
        subspaces.eigenvalues[0],
    )

    covariance = np.cov(phi, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending
    pca_basis = eigenvectors[:, :blind_dim]  # lowest-variance directions

    if calibration_reference is geometry_reference:
        calibration_phi = phi
        logger.warning(
            "calibrating thresholds on the geometry split (legacy same-split "
            "protocol); thresholds will be optimistic"
        )
    else:
        calibration_phi, _, _ = extract_activations(
            model, calibration_reference, device
        )
    logger.info(
        "reference split: geometry n=%d, calibration n=%d",
        phi.shape[0], calibration_phi.shape[0],
    )

    config = MonitorConfig(
        batch_size=batch_size, n_bootstrap=n_bootstrap, quantile=quantile, seed=seed
    )
    return MonitorBundle(
        feather=SubspaceDriftMonitor(subspaces.blind_basis, calibration_phi, config),
        pca=SubspaceDriftMonitor(pca_basis, calibration_phi, config),
        blind_dim=blind_dim,
        fisher_eigenvalues=subspaces.eigenvalues,
        geometry_n=int(phi.shape[0]),
        calibration_n=int(calibration_phi.shape[0]),
    )


def _result_columns(prefix: str, result: MonitorResult) -> dict:
    return {
        f"{prefix}_direction_ratio": round(result.direction_ratio, 6),
        f"{prefix}_shift_magnitude": round(result.shift_magnitude, 6),
        f"{prefix}_energy": round(result.energy, 6),
        f"{prefix}_shift_alarm": int(result.shift_alarm),
        f"{prefix}_energy_alarm": int(result.energy_alarm),
        f"{prefix}_alarm": int(result.alarm),
    }


@torch.no_grad()
def run_episode(
    model: nn.Module,
    bundle: MonitorBundle,
    dataset: Dataset,
    device: torch.device,
    episode: str,
    batch_size: int = 500,
) -> list[dict]:
    """Stream one episode; return one record per batch (labels: eval only)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    weight, bias = head_params(model)
    records = []
    for index, (inputs, targets) in enumerate(loader):
        phi = model.features(inputs.to(device)).cpu().numpy()
        logits = phi @ weight.T + bias
        probabilities = _softmax(logits)
        predictions = probabilities.argmax(axis=1)
        targets_np = targets.numpy()
        entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)

        record = {
            "episode": episode,
            "batch": index,
            "n": len(targets_np),
            "accuracy": round(float((predictions == targets_np).mean()), 6),
            "mean_confidence": round(float(probabilities.max(axis=1).mean()), 6),
            "mean_entropy": round(float(entropy.mean()), 6),
        }
        record.update(_result_columns("feather", bundle.feather.score(phi)))
        record.update(_result_columns("pca", bundle.pca.score(phi)))
        records.append(record)
    return records
