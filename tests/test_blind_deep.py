"""Tests for the deep-feature silent-drift experiment (NumPy only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))

from blind_subspace_deep import output_stats, sensitive_direction, softmax


def synthetic_head(d=32, classes=6, seed=0):
    """A (W, b) and a blind direction u with W u proportional to 1."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((classes, d))
    b = rng.standard_normal(classes)
    # a blind direction: any u with W u in span{1}. Solve W u = 1 (least squares).
    u = np.linalg.lstsq(W, np.ones(classes), rcond=None)[0]
    u = u / np.linalg.norm(u)
    return W, b, u


class TestSoftmax:
    def test_rows_sum_to_one(self):
        p = softmax(np.random.default_rng(0).standard_normal((10, 5)))
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_shift_invariance(self):
        logits = np.random.default_rng(1).standard_normal((4, 5))
        assert np.allclose(softmax(logits), softmax(logits + 3.7))


class TestBlindDirectionIsOutputInvisible:
    def test_blind_shift_leaves_outputs_frozen(self):
        W, b, u = synthetic_head()
        rng = np.random.default_rng(2)
        phi = rng.standard_normal((100, W.shape[1]))
        base = output_stats(phi, W, b)
        shifted = output_stats(phi + 5.0 * u, W, b)
        assert np.abs(shifted["prob"] - base["prob"]).max() < 1e-6
        assert abs(shifted["conf"].mean() - base["conf"].mean()) < 1e-7
        assert abs(shifted["entropy"].mean() - base["entropy"].mean()) < 1e-7

    def test_sensitive_shift_moves_outputs(self):
        W, b, u = synthetic_head()
        # a basis of the blind subspace here is just span{u}; sensitive complement
        blind_basis = u.reshape(-1, 1)
        v = sensitive_direction(blind_basis, seed=3)
        rng = np.random.default_rng(4)
        phi = rng.standard_normal((100, W.shape[1]))
        base = output_stats(phi, W, b)
        shifted = output_stats(phi + 5.0 * v, W, b)
        assert np.abs(shifted["prob"] - base["prob"]).max() > 0.05


class TestSensitiveDirection:
    def test_orthogonal_to_blind_subspace(self):
        rng = np.random.default_rng(5)
        blind_basis, _ = np.linalg.qr(rng.standard_normal((32, 20)))
        v = sensitive_direction(blind_basis, seed=6)
        assert np.allclose(blind_basis.T @ v, 0.0, atol=1e-10)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-10
