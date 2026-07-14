"""Activation-space Fisher information and the sensitive/blind subspace split.

For a classifier with softmax head z = Wφ + b, the Fisher information of the
log-likelihood with respect to the penultimate activation φ is

    F = E[ g gᵀ ],   g = ∇_φ log p(y|φ) = Wᵀ(e_y − p),

estimated empirically on a labeled held-out set (the offline phase). Every
gradient lies in row(W), so rank(F) ≤ C: the top eigenvectors span the
*sensitive* subspace (movement there changes logits) and the remaining
eigenvectors span the *blind* subspace, which contains null(W) — movement
there provably cannot change any model output (docs/revised-plan.md §3,
Proposition 1). F is d×d for activation dimension d, so it is computed
exactly — no KFAC approximation needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_REL_TOL = 1e-8


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def activation_fisher(
    phi: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Compute the empirical activation-space Fisher matrix.

    Args:
        phi: Penultimate activations, shape (n, d).
        y: Integer class labels, shape (n,), values in [0, C).
        weight: Softmax head weight, shape (C, d).
        bias: Softmax head bias, shape (C,).

    Returns:
        The d×d empirical Fisher matrix (symmetric positive semi-definite).

    Raises:
        ValueError: On shape mismatches or out-of-range labels.
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y)
    weight = np.asarray(weight, dtype=float)
    bias = np.asarray(bias, dtype=float)

    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D (n, d), got shape {phi.shape}")
    n, d = phi.shape
    if weight.ndim != 2 or weight.shape[1] != d:
        raise ValueError(f"weight must have shape (C, {d}), got {weight.shape}")
    n_classes = weight.shape[0]
    if bias.shape != (n_classes,):
        raise ValueError(f"bias must have shape ({n_classes},), got {bias.shape}")
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError(f"labels must lie in [0, {n_classes})")

    probabilities = _softmax(phi @ weight.T + bias)
    residual = -probabilities
    residual[np.arange(n), y] += 1.0  # e_y − p
    gradients = residual @ weight  # (n, d), each row Wᵀ(e_y − p)
    return gradients.T @ gradients / n


@dataclass(frozen=True)
class FisherSubspaces:
    """Eigendecomposition of the Fisher matrix, split into two subspaces.

    Attributes:
        eigenvalues: All eigenvalues, sorted descending.
        sensitive_basis: Orthonormal columns spanning the high-Fisher
            (output-sensitive) subspace, shape (d, k).
        blind_basis: Orthonormal columns spanning the near-null
            (output-blind) subspace, shape (d, d−k).
    """

    eigenvalues: np.ndarray
    sensitive_basis: np.ndarray
    blind_basis: np.ndarray


def fisher_subspaces(
    fisher: np.ndarray, rel_tol: float = DEFAULT_REL_TOL
) -> FisherSubspaces:
    """Split activation space into sensitive and blind Fisher subspaces.

    Args:
        fisher: Symmetric d×d Fisher matrix.
        rel_tol: Eigenvalues below ``rel_tol * λ_max`` are treated as blind.

    Returns:
        A FisherSubspaces with descending eigenvalues and orthonormal bases.

    Raises:
        ValueError: If ``fisher`` is not a symmetric square matrix.
    """
    fisher = np.asarray(fisher, dtype=float)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"fisher must be square, got shape {fisher.shape}")
    if not np.allclose(fisher, fisher.T):
        raise ValueError("fisher must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    cutoff = rel_tol * max(eigenvalues[0], 0.0)
    n_sensitive = int(np.sum(eigenvalues > cutoff))
    return FisherSubspaces(
        eigenvalues=eigenvalues,
        sensitive_basis=eigenvectors[:, :n_sensitive],
        blind_basis=eigenvectors[:, n_sensitive:],
    )
