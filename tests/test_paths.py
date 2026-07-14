"""Tests for dataset path resolution (no torch or data required)."""

import pytest

from feather.data.paths import DATA_ENV, cifar10c_dir, data_root, mnist_root


class TestDataRoot:
    def test_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_ENV, "C:/somewhere-else")
        assert data_root(tmp_path) == tmp_path

    def test_env_var_used(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_ENV, str(tmp_path))
        assert data_root() == tmp_path

    def test_default_is_local_data(self, monkeypatch):
        monkeypatch.delenv(DATA_ENV, raising=False)
        assert str(data_root()) == "data"


class TestHelpfulErrors:
    def test_missing_mnist_mentions_docs(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="docs/datasets.md"):
            mnist_root(tmp_path)

    def test_missing_cifar10c_mentions_env_var(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=DATA_ENV):
            cifar10c_dir(tmp_path)
