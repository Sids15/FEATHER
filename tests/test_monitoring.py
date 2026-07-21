"""Tests for the torch<->numpy monitoring bridge (run where torch exists)."""

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset

from feather.models import SmallCNN
from feather.monitoring import (
    extract_activations,
    fit_monitors,
    head_params,
    load_frozen_model,
    run_episode,
    split_reference_dataset,
)

DEVICE = torch.device("cpu")


def fake_mnist(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return TensorDataset(x, y)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SmallCNN().eval()


class TestBridge:
    def test_extract_activations_shapes(self, model):
        phi, logits, labels = extract_activations(model, fake_mnist(), DEVICE)
        assert phi.shape == (256, 128)
        assert logits.shape == (256, 10)
        assert labels.shape == (256,)

    def test_head_params_match_model(self, model):
        weight, bias = head_params(model)
        assert weight.shape == (10, 128)
        assert bias.shape == (10,)

    def test_load_frozen_model_roundtrip(self, model, tmp_path):
        path = tmp_path / "final_model.pt"
        torch.save({"model_state": model.state_dict(), "model_meta": model.meta()}, path)
        loaded = load_frozen_model(path, DEVICE)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x))
        assert not any(p.requires_grad for p in loaded.parameters())

    def test_load_rejects_unknown_arch(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"model_state": {}, "model_meta": {"arch": "mystery"}}, path)
        with pytest.raises(ValueError, match="architecture"):
            load_frozen_model(path, DEVICE)


class TestSplitReferenceDataset:
    def test_sizes_and_disjoint(self):
        geometry, calibration = split_reference_dataset(fake_mnist(200), 0.5, seed=0)
        assert len(geometry) == 100 and len(calibration) == 100
        assert set(geometry.indices).isdisjoint(calibration.indices)
        assert sorted(geometry.indices + calibration.indices) == list(range(200))

    def test_deterministic_given_seed(self):
        dataset = fake_mnist(200)
        first = split_reference_dataset(dataset, 0.5, seed=3)
        second = split_reference_dataset(dataset, 0.5, seed=3)
        assert first[0].indices == second[0].indices
        assert first[1].indices == second[1].indices

    def test_different_seeds_differ(self):
        dataset = fake_mnist(200)
        assert (
            split_reference_dataset(dataset, 0.5, seed=0)[0].indices
            != split_reference_dataset(dataset, 0.5, seed=1)[0].indices
        )

    def test_rejects_degenerate_fraction(self):
        with pytest.raises(ValueError, match="fraction"):
            split_reference_dataset(fake_mnist(10), 1.5)
        with pytest.raises(ValueError, match="empty"):
            split_reference_dataset(fake_mnist(10), 0.001)


class TestMonitoringPipeline:
    def test_fit_and_episode_records(self, model):
        geometry, calibration = split_reference_dataset(fake_mnist(), 0.5, seed=0)
        bundle = fit_monitors(
            model, geometry, calibration, DEVICE, batch_size=64, n_bootstrap=50
        )
        assert 0 < bundle.blind_dim < 128  # rank <= C-1 = 9 sensitive dims
        assert bundle.blind_dim >= 128 - 9
        assert bundle.geometry_n == 128 and bundle.calibration_n == 128

        records = run_episode(
            model, bundle, fake_mnist(seed=1), DEVICE, episode="e0", batch_size=64
        )
        assert len(records) == 4
        first = records[0]
        for column in (
            "episode", "batch", "accuracy", "mean_confidence", "mean_entropy",
            "atc_score", "atc_alarm", "conf_score", "conf_alarm",
            "entropy_score", "entropy_alarm", "proxy_score", "proxy_alarm",
            "feather_shift_magnitude", "feather_alarm",
            "pca_shift_magnitude", "pca_alarm",
        ):
            assert column in first
        assert 0.0 <= first["accuracy"] <= 1.0

    def test_geometry_fixed_thresholds_follow_calibration(self, model):
        """Basis comes from the geometry split, thresholds from calibration."""
        geometry = fake_mnist(seed=0)
        bundle_a = fit_monitors(
            model, geometry, fake_mnist(seed=1), DEVICE, batch_size=64, n_bootstrap=50
        )
        bundle_b = fit_monitors(
            model, geometry, fake_mnist(seed=2), DEVICE, batch_size=64, n_bootstrap=50
        )
        assert (bundle_a.fisher_eigenvalues == bundle_b.fisher_eigenvalues).all()
        assert (bundle_a.feather._basis == bundle_b.feather._basis).all()
        assert bundle_a.feather.shift_threshold != bundle_b.feather.shift_threshold
        assert (bundle_a.feather._mu_ref != bundle_b.feather._mu_ref).any()

    def test_run_episode_saves_raw_logits(self, model, tmp_path):
        import numpy as np

        geometry, calibration = split_reference_dataset(fake_mnist(), 0.5, seed=0)
        bundle = fit_monitors(
            model, geometry, calibration, DEVICE, batch_size=64, n_bootstrap=50
        )
        raw_path = tmp_path / "raw" / "episode_e0.npz"
        records = run_episode(
            model, bundle, fake_mnist(seed=1), DEVICE, episode="e0",
            batch_size=64, raw_path=raw_path,
        )
        raw = np.load(raw_path)
        assert raw["logits"].shape == (256, 10)
        assert raw["labels"].shape == (256,)
        assert sum(r["n"] for r in records) == 256

    def test_heldout_differs_from_same_split(self, model):
        geometry = fake_mnist(seed=0)
        same_split = fit_monitors(
            model, geometry, geometry, DEVICE, batch_size=64, n_bootstrap=50
        )
        heldout = fit_monitors(
            model, geometry, fake_mnist(seed=1), DEVICE, batch_size=64, n_bootstrap=50
        )
        assert same_split.feather.shift_threshold != heldout.feather.shift_threshold
