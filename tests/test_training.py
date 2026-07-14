"""Tests for the resumable trainer (run on a machine with PyTorch installed).

Uses a tiny synthetic dataset so no real data or GPU is required. The whole
file is skipped automatically where torch is unavailable (this laptop).
"""

import json

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import TensorDataset

from feather.training import TrainConfig, Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 8)
        self.head = nn.Linear(8, 2)

    def features(self, x):
        return torch.relu(self.backbone(x))

    def forward(self, x):
        return self.head(self.features(x))

    def meta(self):
        return {"arch": "tiny"}


def tiny_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x.sum(dim=1) > 0).long()
    return TensorDataset(x, y)


def make_config(tmp_path, run_name="run", epochs=2, **overrides):
    defaults = dict(
        run_name=run_name,
        epochs=epochs,
        seed=0,
        batch_size=16,
        optimizer="adam",
        lr=1e-2,
        weight_decay=0.0,
        scheduler="none",
        num_workers=0,
        device="cpu",
        output_root=tmp_path / "outputs",
        log_root=tmp_path / "logs",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def make_trainer(tmp_path, **overrides):
    return Trainer(TinyModel(), tiny_data(), tiny_data(seed=1), make_config(tmp_path, **overrides))


class TestOutputs:
    def test_run_artifacts_written(self, tmp_path):
        trainer = make_trainer(tmp_path)
        summary = trainer.fit()
        run_dir = tmp_path / "outputs" / "run"
        for name in ("config.json", "env.json", "metrics.csv", "summary.json", "final_model.pt"):
            assert (run_dir / name).exists(), name
        assert (tmp_path / "logs" / "run.log").exists()
        assert summary["epochs_trained"] == 2

    def test_metrics_csv_has_one_row_per_epoch(self, tmp_path):
        trainer = make_trainer(tmp_path, epochs=3)
        trainer.fit()
        lines = (tmp_path / "outputs" / "run" / "metrics.csv").read_text().strip().splitlines()
        assert len(lines) == 1 + 3  # header + epochs
        assert lines[0].startswith("epoch,train_loss,train_acc,val_loss,val_acc,lr")

    def test_summary_json_matches_return(self, tmp_path):
        trainer = make_trainer(tmp_path)
        summary = trainer.fit()
        on_disk = json.loads((tmp_path / "outputs" / "run" / "summary.json").read_text())
        assert on_disk == summary


class TestCheckpointing:
    def test_checkpoints_saved_and_pruned(self, tmp_path):
        trainer = make_trainer(tmp_path, epochs=5, keep_last=2)
        trainer.fit()
        ckpts = sorted((tmp_path / "outputs" / "run" / "checkpoints").glob("epoch_*.pt"))
        assert [c.name for c in ckpts] == ["epoch_0004.pt", "epoch_0005.pt"]
        assert (tmp_path / "outputs" / "run" / "checkpoints" / "best.pt").exists()

    def test_checkpoint_is_weights_only_loadable(self, tmp_path):
        trainer = make_trainer(tmp_path)
        trainer.fit()
        path = tmp_path / "outputs" / "run" / "checkpoints" / "epoch_0002.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["epoch"] == 2

    def test_resume_continues_to_target_epochs(self, tmp_path):
        make_trainer(tmp_path, epochs=2).fit()
        resumed = make_trainer(tmp_path, epochs=4)
        summary = resumed.fit(resume="auto")
        assert resumed.start_epoch == 3
        assert summary["epochs_trained"] == 4
        lines = (tmp_path / "outputs" / "run" / "metrics.csv").read_text().strip().splitlines()
        assert len(lines) == 1 + 4

    def test_resume_restores_weights_exactly(self, tmp_path):
        first = make_trainer(tmp_path, epochs=2)
        first.fit()
        resumed = make_trainer(tmp_path, epochs=2)
        resumed.load_checkpoint("auto")
        for a, b in zip(first.model.parameters(), resumed.model.parameters()):
            assert torch.equal(a, b)

    def test_resume_auto_without_checkpoints_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            make_trainer(tmp_path).fit(resume="auto")


class TestReproducibility:
    def test_same_seed_same_first_epoch(self, tmp_path):
        s1 = make_trainer(tmp_path, run_name="a", epochs=1).fit()
        s2 = make_trainer(tmp_path, run_name="b", epochs=1).fit()
        assert s1["final_val_acc"] == s2["final_val_acc"]


class TestConfigValidation:
    def test_bad_values_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            make_config(tmp_path, epochs=0)
        with pytest.raises(ValueError):
            make_config(tmp_path, run_name="bad name")
        with pytest.raises(ValueError):
            make_config(tmp_path, optimizer="rmsprop")
