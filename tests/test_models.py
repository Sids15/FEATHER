"""Tests for model architectures (run where PyTorch is installed)."""

import pytest

torch = pytest.importorskip("torch")

from feather.models import CifarResNet18, SmallCNN


class TestSmallCNN:
    def test_shapes(self):
        model = SmallCNN()
        x = torch.randn(4, 1, 28, 28)
        assert model(x).shape == (4, 10)
        assert model.features(x).shape == (4, 128)

    def test_logits_equal_head_of_features(self):
        """FEATHER reads (features, head): forward must be exactly head(features)."""
        model = SmallCNN().eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            assert torch.allclose(model(x), model.head(model.features(x)))


class TestCifarResNet18:
    def test_shapes(self):
        model = CifarResNet18()
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 10)
        assert model.features(x).shape == (2, 512)

    def test_logits_equal_head_of_features(self):
        model = CifarResNet18().eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert torch.allclose(model(x), model.head(model.features(x)), atol=1e-6)

    def test_meta_serializable(self):
        import json

        json.dumps(CifarResNet18().meta())
        json.dumps(SmallCNN().meta())
