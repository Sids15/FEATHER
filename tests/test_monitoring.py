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


class TestMonitoringPipeline:
    def test_fit_and_episode_records(self, model):
        bundle = fit_monitors(
            model, fake_mnist(), DEVICE, batch_size=64, n_bootstrap=50
        )
        assert 0 < bundle.blind_dim < 128  # rank <= C-1 = 9 sensitive dims
        assert bundle.blind_dim >= 128 - 9

        records = run_episode(
            model, bundle, fake_mnist(seed=1), DEVICE, episode="e0", batch_size=64
        )
        assert len(records) == 4
        first = records[0]
        for column in (
            "episode", "batch", "accuracy", "mean_confidence", "mean_entropy",
            "feather_shift_magnitude", "feather_alarm",
            "pca_shift_magnitude", "pca_alarm",
        ):
            assert column in first
        assert 0.0 <= first["accuracy"] <= 1.0
