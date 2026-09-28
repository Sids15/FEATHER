"""WILDS Camelyon17 loader + ResNet-18 interface checks.

Torch-dependent, and the data tests also need the ``wilds`` package and the
downloaded dataset, so everything skips cleanly off the workstation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from feather.models import ResNet18


def _load_or_skip(fn):
    try:
        return fn()
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        pytest.skip(f"Camelyon17 data not available: {exc}")


def test_resnet18_interface():
    model = ResNet18(num_classes=2, pretrained=False).eval()
    x = torch.randn(2, 3, 96, 96)
    phi = model.features(x)
    assert tuple(phi.shape) == (2, 512)
    assert model.head.out_features == 2
    assert model.meta() == {"arch": "resnet18", "num_classes": 2}
    assert tuple(model(x).shape) == (2, 2)


def test_camelyon17_reference_shapes():
    pytest.importorskip("wilds")
    from feather.data.wilds import camelyon17_reference

    reference = _load_or_skip(lambda: camelyon17_reference(max_samples=64))
    assert len(reference) > 0
    image, label = reference[0]
    assert tuple(image.shape) == (3, 96, 96)
    assert label in (0, 1)


def test_camelyon17_episodes_present_and_labeled():
    pytest.importorskip("wilds")
    from feather.data.wilds import camelyon17_episodes

    episodes = _load_or_skip(lambda: list(camelyon17_episodes(max_samples=64)))
    names = [name for name, _ in episodes]
    assert names[0] == "clean"  # in-distribution episode comes first
    assert "ood_test_hosp2" in names
    for _name, dataset in episodes:
        assert len(dataset) > 0
        _image, label = dataset[0]
        assert label in (0, 1)
