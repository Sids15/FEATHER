"""Tests for the numpy-only aggregation helpers in analyze_monitoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))

from analyze_monitoring import auroc, paired_sign_flip_test, per_seed_alarm_rates

import numpy as np


def episode_row(seed, episode, label, feather_rate, pca_rate):
    return {
        "seed": seed,
        "episode": episode,
        "label": label,
        "feather_alarm_rate": feather_rate,
        "pca_alarm_rate": pca_rate,
    }


class TestPerSeedAlarmRates:
    def rows(self):
        out = []
        for seed in (0, 1):
            offset = 0.1 * seed
            out += [
                episode_row(seed, "clean", "benign", 0.0 + offset, 0.1 + offset),
                episode_row(seed, "fog_s1", "benign", 0.2 + offset, 0.4 + offset),
                episode_row(seed, "fog_s3", "gray", 0.5 + offset, 0.6 + offset),
                episode_row(seed, "fog_s5", "harmful", 1.0, 1.0),
            ]
        return out

    def test_clean_excluded_from_benign(self):
        summary = per_seed_alarm_rates(self.rows(), "feather", "clean")
        assert summary["clean"]["per_seed"] == {"0": 0.0, "1": 0.1}
        # benign covers only fog_s1, not the clean episode
        assert summary["benign"]["per_seed"] == {"0": 0.2, "1": 0.3}

    def test_mean_std_across_seeds(self):
        summary = per_seed_alarm_rates(self.rows(), "pca", "clean")
        assert summary["benign"]["mean"] == 0.45
        assert summary["benign"]["std"] == 0.05
        assert summary["harmful"]["mean"] == 1.0 and summary["harmful"]["std"] == 0.0

    def test_empty_category_is_none(self):
        rows = [episode_row(0, "clean", "benign", 0.0, 0.0)]
        summary = per_seed_alarm_rates(rows, "feather", "clean")
        assert summary["harmful"]["mean"] is None
        assert summary["harmful"]["per_seed"] == {}


class TestPairedSignFlipTest:
    def test_consistent_gap_gets_smallest_exact_p(self):
        result = paired_sign_flip_test([0.3, 0.35, 0.32, 0.31, 0.34])
        # all 5 diffs share a sign: only the 2 all-same-sign flips of 32 match
        assert result["p_value"] == round(2 / 32, 4)
        assert result["mean_difference"] == round(np.mean([0.3, 0.35, 0.32, 0.31, 0.34]), 4)

    def test_mixed_signs_not_significant(self):
        result = paired_sign_flip_test([0.1, -0.1, 0.05, -0.05, 0.0])
        assert result["p_value"] > 0.5

    def test_reports_per_seed_differences(self):
        diffs = [0.2, 0.1, 0.3]
        assert paired_sign_flip_test(diffs)["per_seed_differences"] == diffs


class TestAuroc:
    def test_perfect_separation(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert auroc(scores, labels) == 1.0

    def test_reversed_separation(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        assert auroc(scores, labels) == 0.0

    def test_single_class_is_nan(self):
        assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))
