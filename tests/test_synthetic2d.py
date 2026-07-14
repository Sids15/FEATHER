"""Tests for the Tier-1 2D synthetic drift stream (docs/revised-plan.md §4).

The generator must provide streams with ground-truth drift labels by
construction: shifts parallel to the Bayes decision boundary are benign
(accuracy preserved), shifts along the boundary normal are harmful
(accuracy degrades toward chance).
"""

import numpy as np
import pytest

from feather.data.synthetic2d import (
    DriftSpec,
    Synthetic2DConfig,
    Synthetic2DStream,
    bayes_predict,
    benign_drift,
    harmful_drift,
)


def collect(stream: Synthetic2DStream) -> list:
    return list(stream)


def default_config(**overrides) -> Synthetic2DConfig:
    defaults = dict(batch_size=500, n_batches=20, seed=123)
    defaults.update(overrides)
    return Synthetic2DConfig(**defaults)


class TestDeterminism:
    def test_same_seed_gives_identical_batches(self):
        drift = DriftSpec(kind="abrupt", direction=(1.0, 0.0), magnitude=2.0, onset=10)
        a = collect(Synthetic2DStream(default_config(), drift))
        b = collect(Synthetic2DStream(default_config(), drift))
        for ba, bb in zip(a, b):
            np.testing.assert_array_equal(ba.x, bb.x)
            np.testing.assert_array_equal(ba.y, bb.y)

    def test_different_seeds_give_different_batches(self):
        drift = DriftSpec(kind="none")
        a = collect(Synthetic2DStream(default_config(seed=1), drift))
        b = collect(Synthetic2DStream(default_config(seed=2), drift))
        assert not np.array_equal(a[0].x, b[0].x)


class TestBatchStructure:
    def test_shapes_labels_and_count(self):
        cfg = default_config()
        batches = collect(Synthetic2DStream(cfg, DriftSpec(kind="none")))
        assert len(batches) == cfg.n_batches
        for i, batch in enumerate(batches):
            assert batch.index == i
            assert batch.x.shape == (cfg.batch_size, 2)
            assert batch.y.shape == (cfg.batch_size,)
            assert set(np.unique(batch.y)) == {0, 1}


class TestDriftInjection:
    def test_no_shift_before_onset(self):
        drift = DriftSpec(kind="abrupt", direction=(1.0, 0.0), magnitude=3.0, onset=10)
        batches = collect(Synthetic2DStream(default_config(), drift))
        for batch in batches[:10]:
            np.testing.assert_array_equal(batch.applied_shift, np.zeros(2))

    def test_abrupt_full_shift_from_onset(self):
        drift = DriftSpec(kind="abrupt", direction=(0.0, 1.0), magnitude=3.0, onset=10)
        batches = collect(Synthetic2DStream(default_config(), drift))
        for batch in batches[10:]:
            np.testing.assert_allclose(batch.applied_shift, [0.0, 3.0])

    def test_abrupt_shift_moves_empirical_mean(self):
        cfg = default_config(batch_size=4000)
        drift = DriftSpec(kind="abrupt", direction=(0.0, 1.0), magnitude=3.0, onset=10)
        batches = collect(Synthetic2DStream(cfg, drift))
        pre_mean_y = np.mean([b.x[:, 1].mean() for b in batches[:10]])
        post_mean_y = np.mean([b.x[:, 1].mean() for b in batches[10:]])
        assert post_mean_y - pre_mean_y == pytest.approx(3.0, abs=0.15)

    def test_gradual_shift_ramps_to_full_magnitude(self):
        drift = DriftSpec(
            kind="gradual", direction=(1.0, 0.0), magnitude=2.0, onset=5, duration=10
        )
        batches = collect(Synthetic2DStream(default_config(), drift))
        norms = [float(np.linalg.norm(b.applied_shift)) for b in batches]
        assert norms[4] == 0.0
        assert all(b >= a for a, b in zip(norms, norms[1:]))  # nondecreasing
        assert norms[15] == pytest.approx(2.0)
        assert 0.0 < norms[9] < 2.0  # mid-ramp is partial

    def test_direction_is_normalized_internally(self):
        drift = DriftSpec(kind="abrupt", direction=(2.0, 0.0), magnitude=1.0, onset=0)
        batches = collect(Synthetic2DStream(default_config(), drift))
        np.testing.assert_allclose(batches[0].applied_shift, [1.0, 0.0])


class TestValidation:
    def test_zero_direction_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            DriftSpec(kind="abrupt", direction=(0.0, 0.0), magnitude=1.0, onset=0)

    def test_negative_magnitude_rejected(self):
        with pytest.raises(ValueError, match="magnitude"):
            DriftSpec(kind="abrupt", direction=(1.0, 0.0), magnitude=-1.0, onset=0)

    def test_drifting_kind_requires_direction(self):
        with pytest.raises(ValueError, match="direction"):
            DriftSpec(kind="abrupt", magnitude=1.0, onset=0)

    def test_gradual_requires_positive_duration(self):
        with pytest.raises(ValueError, match="duration"):
            DriftSpec(
                kind="gradual", direction=(1.0, 0.0), magnitude=1.0, onset=0, duration=0
            )


class TestGroundTruthHarmfulness:
    """The core Tier-1 property: harmfulness is known by construction."""

    def accuracy(self, cfg, batches):
        accs = [np.mean(bayes_predict(b.x, cfg) == b.y) for b in batches]
        return float(np.mean(accs))

    def test_clean_stream_accuracy_is_high(self):
        cfg = default_config()
        batches = collect(Synthetic2DStream(cfg, DriftSpec(kind="none")))
        assert self.accuracy(cfg, batches) > 0.9

    def test_benign_drift_preserves_accuracy(self):
        cfg = default_config()
        batches = collect(Synthetic2DStream(cfg, benign_drift(cfg, magnitude=5.0, onset=5)))
        assert self.accuracy(cfg, batches[5:]) > 0.9

    def test_harmful_drift_degrades_accuracy(self):
        cfg = default_config()
        batches = collect(Synthetic2DStream(cfg, harmful_drift(cfg, magnitude=3.0, onset=5)))
        assert self.accuracy(cfg, batches[5:]) < 0.75
