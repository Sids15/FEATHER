"""Tests for the activation-space Fisher matrix and subspace split.

For a softmax head z = Wφ + b, the per-sample Fisher gradient is
g = Wᵀ(e_y − p), so every gradient lies in row(W): rank(F) ≤ C and the
blind subspace contains null(W) — movement there cannot change logits
(Detector Blindness, docs/revised-plan.md §3 Prop. 1).
"""

import numpy as np
import pytest

from feather.core.fisher import activation_fisher, fisher_subspaces


def make_binary_head(w=(2.0, -1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Binary softmax head equivalent to logistic regression with weights w."""
    weight = np.array([[0.0, 0.0], list(w)])
    bias = np.zeros(2)
    return weight, bias


def random_data(n=400, d=2, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((n, d))
    y = rng.integers(0, n_classes, size=n)
    return phi, y


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TestActivationFisher:
    def test_shape_symmetry_and_psd(self):
        phi, y = random_data()
        weight, bias = make_binary_head()
        fisher = activation_fisher(phi, y, weight, bias)
        assert fisher.shape == (2, 2)
        np.testing.assert_allclose(fisher, fisher.T)
        assert np.all(np.linalg.eigvalsh(fisher) >= -1e-12)

    def test_binary_head_fisher_is_rank_one_along_w(self):
        phi, y = random_data()
        weight, bias = make_binary_head(w=(2.0, -1.0))
        fisher = activation_fisher(phi, y, weight, bias)
        eigenvalues, eigenvectors = np.linalg.eigh(fisher)
        assert eigenvalues[0] <= 1e-10 * eigenvalues[1]  # rank one
        top = eigenvectors[:, 1]
        w_unit = np.array([2.0, -1.0]) / np.sqrt(5.0)
        assert abs(top @ w_unit) == pytest.approx(1.0, abs=1e-8)

    def test_shape_mismatch_rejected(self):
        phi, y = random_data()
        weight, bias = make_binary_head()
        with pytest.raises(ValueError):
            activation_fisher(phi[:, :1], y, weight, bias)
        with pytest.raises(ValueError):
            activation_fisher(phi, y[:-1], weight, bias)

    def test_label_out_of_range_rejected(self):
        phi, y = random_data()
        weight, bias = make_binary_head()
        with pytest.raises(ValueError):
            activation_fisher(phi, y + 5, weight, bias)


class TestFisherSubspaces:
    def build(self):
        phi, y = random_data()
        weight, bias = make_binary_head(w=(2.0, -1.0))
        fisher = activation_fisher(phi, y, weight, bias)
        return fisher, weight, bias, phi

    def test_split_dimensions_and_orthonormality(self):
        fisher, *_ = self.build()
        sub = fisher_subspaces(fisher)
        assert sub.sensitive_basis.shape == (2, 1)
        assert sub.blind_basis.shape == (2, 1)
        basis = np.hstack([sub.sensitive_basis, sub.blind_basis])
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)

    def test_eigenvalues_sorted_descending(self):
        fisher, *_ = self.build()
        sub = fisher_subspaces(fisher)
        assert all(a >= b for a, b in zip(sub.eigenvalues, sub.eigenvalues[1:]))

    def test_blind_direction_leaves_softmax_unchanged(self):
        """Numerical check of Proposition 1 (Detector Blindness)."""
        fisher, weight, bias, phi = self.build()
        sub = fisher_subspaces(fisher)
        blind = sub.blind_basis[:, 0]
        shifted = phi + 5.0 * blind
        p_before = softmax(phi @ weight.T + bias)
        p_after = softmax(shifted @ weight.T + bias)
        np.testing.assert_allclose(p_after, p_before, atol=1e-10)

    def test_sensitive_direction_changes_softmax(self):
        fisher, weight, bias, phi = self.build()
        sub = fisher_subspaces(fisher)
        sensitive = sub.sensitive_basis[:, 0]
        shifted = phi + 5.0 * sensitive
        p_before = softmax(phi @ weight.T + bias)
        p_after = softmax(shifted @ weight.T + bias)
        assert np.abs(p_after - p_before).max() > 0.1

    def test_non_square_matrix_rejected(self):
        with pytest.raises(ValueError):
            fisher_subspaces(np.zeros((2, 3)))
