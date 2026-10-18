"""BREEDS-style CIFAR-100 split and episode construction (NumPy-only parts)."""

import numpy as np
import pytest

from feather.data.breeds import gather, mix_indices, split_subclasses, swap_indices


def fake_labels(rng, per_fine=3):
    # 100 fine ids scattered over 20 superclasses, 5 each (like CIFAR-100)
    coarse_of_fine = rng.permutation(np.repeat(np.arange(20), 5))
    fine = np.repeat(np.arange(100), per_fine)
    return fine, coarse_of_fine[fine]


def test_split_holds_out_one_subclass_per_superclass():
    fine, coarse = fake_labels(np.random.default_rng(0))
    seen, held = split_subclasses(fine, coarse, holdout_rank=4)
    assert len(seen) == 80 and len(held) == 20 and not set(seen) & set(held)
    held_coarse = {int(coarse[fine == h][0]) for h in held}
    assert held_coarse == set(range(20))  # exactly one per superclass
    for h in held:  # rank 4 = the largest fine id within its superclass
        c = coarse[fine == h][0]
        assert h == np.unique(fine[coarse == c]).max()


def test_split_rejects_malformed_superclass():
    fine, coarse = fake_labels(np.random.default_rng(1))
    coarse = coarse.copy()
    coarse[fine == fine[0]] = (coarse[fine == fine[0]][0] + 1) % 20  # 6 subclasses
    with pytest.raises(ValueError):
        split_subclasses(fine, coarse)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 1.0])
def test_mix_has_requested_fraction_without_repeats(rho):
    source, index = mix_indices(800, 1200, 800, rho, np.random.default_rng(2))
    assert len(source) == 800 and source.sum() == round(rho * 800)
    for s in (0, 1):
        assert len(set(index[source == s])) == (source == s).sum()
    if 0 < rho < 1:  # shuffled: the shift share is spread, not front-loaded
        assert abs(source[:400].mean() - rho) < 0.1


def test_swap_keeps_class_mix_and_replaces_one_superclass():
    rng = np.random.default_rng(3)
    clean_c = np.repeat(np.arange(20), 40)
    shift_c = np.repeat(np.arange(20), 60)
    source, index = swap_indices(clean_c, shift_c, superclass=7, rng=rng)
    _, y = gather(source, index, (np.zeros((800, 1)), clean_c), (np.ones((1200, 1)), shift_c))
    assert np.array_equal(np.bincount(y, minlength=20), np.bincount(clean_c))
    assert set(y[source == 1]) == {7} and 7 not in set(y[source == 0])


def test_gather_takes_each_sample_from_its_pool():
    clean = (np.arange(5)[:, None] * 1.0, np.arange(5))
    shift = (np.arange(100, 105)[:, None] * 1.0, np.arange(10, 15))
    x, y = gather(np.array([0, 1, 0]), np.array([4, 2, 0]), clean, shift)
    assert x[:, 0].tolist() == [4.0, 102.0, 0.0] and y.tolist() == [4, 12, 0]
