"""Parity of the pairwise diversity matrix against `deslib.util.diversity`.

The upstream `compute_pairwise_diversity` runs three nested Python loops and
calls a Python function m*(m-1)/2 times, so any of these tests failing means
the Mojo contingency-count sweep disagrees with the reference on a real
measure, not on a rounding detail.
"""

import numpy as np
import pytest

import mojo_deslib as mdes

from deslib.util import diversity as dv

MEASURES = [
    (mdes.DOUBLE_FAULT, dv.double_fault),
    (mdes.NEGATIVE_DOUBLE_FAULT, dv.negative_double_fault),
    (mdes.Q_STATISTIC, dv.Q_statistic),
    (mdes.RATIO_ERRORS, dv.ratio_errors),
    (mdes.DISAGREEMENT, dv.disagreement_measure),
    (mdes.AGREEMENT, dv.agreement_measure),
    (mdes.CORRELATION, dv.correlation_coefficient),
]

IDS = [name for name, _ in MEASURES]


def _labels(rng, n, m, n_classes):
    """A pool that is better than chance, so Q and rho have a real spread."""
    y = rng.integers(0, n_classes, size=n)
    pred = np.empty((n, m), dtype=np.int64)
    for j in range(m):
        flip = rng.random(n) < 0.15 + 0.05 * j
        noise = rng.integers(0, n_classes, size=n)
        col = np.where(rng.random(n) < 0.6, y, noise)
        pred[:, j] = np.where(flip, (col + 1) % n_classes, col)
    return y, pred


@pytest.mark.parametrize("code,func", MEASURES, ids=IDS)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_diversity_matches_deslib(code, func, n_classes):
    rng = np.random.default_rng(1000 + code * 10 + n_classes)
    y, pred = _labels(rng, 400, 9, n_classes)
    got = mdes.compute_pairwise_diversity(y, pred, code)
    want = dv.compute_pairwise_diversity(y, pred, func)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("code,func", MEASURES, ids=IDS)
def test_diversity_degenerate_no_double_errors(code, func):
    """Two classifiers that never err together drive N00 to 0.

    `ratio_errors` answers `sys.float_info.max` there and the others divide by
    a zero numerator, so this pins the branch that only a naive
    `(N01 + N10) / N00` would get wrong: a crash, or an infinity.
    """
    rng = np.random.default_rng(77)
    y = rng.integers(0, 2, size=200)
    pred = np.empty((200, 4), dtype=np.int64)
    pred[:, 0] = y
    pred[:, 1] = y
    pred[:, 2] = 1 - y
    pred[:, 3] = rng.integers(0, 2, size=200)
    got = mdes.compute_pairwise_diversity(y, pred, code)
    if code == mdes.Q_STATISTIC:
        # The two perfect columns are both-correct on every sample, so
        # N11*N00 + N01*N10 is 0 and deslib itself raises ZeroDivisionError.
        # The kernel returns nan there; this is the one documented behavioural
        # difference in the diversity kernel.
        with pytest.raises(ZeroDivisionError):
            dv.compute_pairwise_diversity(y, pred, func)
        assert np.all(np.isnan(got))
        return
    want = dv.compute_pairwise_diversity(y, pred, func)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_ratio_errors_returns_float_max_when_n00_zero():
    """The sentinel is exact: a kernel returning +inf would not match."""
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    pred = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    got = mdes.compute_pairwise_diversity(y, pred, mdes.RATIO_ERRORS)
    assert got[0] == np.finfo(np.float64).max
    assert got[1] == np.finfo(np.float64).max


def test_diversity_depends_on_classifier_column_order():
    """A transposed prediction matrix would silently give the wrong answer.

    Reversing the classifier axis permutes the pair list, so the accumulated
    vector must change; a kernel reading the (n, m) buffer with the wrong
    stride would return the same vector for both orderings.
    """
    rng = np.random.default_rng(5)
    y, pred = _labels(rng, 300, 6, 3)
    straight = mdes.compute_pairwise_diversity(y, pred, mdes.Q_STATISTIC)
    flipped = mdes.compute_pairwise_diversity(y, pred[:, ::-1], mdes.Q_STATISTIC)
    assert not np.allclose(straight, flipped)
    np.testing.assert_allclose(
        straight, flipped[::-1], rtol=1e-12, atol=1e-12
    )


def test_diversity_single_classifier_is_all_zero():
    rng = np.random.default_rng(11)
    y, pred = _labels(rng, 50, 1, 3)
    got = mdes.compute_pairwise_diversity(y, pred, mdes.Q_STATISTIC)
    assert got.shape == (1,)
    assert got[0] == 0.0


def test_q_statistic_is_bounded_by_one():
    """Sanity anchor that does not depend on deslib being installed correctly."""
    rng = np.random.default_rng(3)
    y, pred = _labels(rng, 500, 8, 2)
    q = mdes.compute_pairwise_diversity(y, pred, mdes.Q_STATISTIC)
    assert np.all(np.abs(q) <= 1.0 + 1e-12)


def test_agreement_and_disagreement_complement_pairwise():
    """N00 + N10 + N01 + N11 == 1 for every pair, so summing both measures
    over the pool must give exactly twice the number of pairs."""
    rng = np.random.default_rng(21)
    y, pred = _labels(rng, 250, 5, 3)
    m = pred.shape[1]
    n_pairs = m * (m - 1) // 2
    dis = mdes.compute_pairwise_diversity(y, pred, mdes.DISAGREEMENT)
    agr = mdes.compute_pairwise_diversity(y, pred, mdes.AGREEMENT)
    assert dis.sum() + agr.sum() == pytest.approx(2.0 * n_pairs, rel=1e-12)


def test_diversity_sum_equals_two_times_pairwise_total():
    """The accumulator must add each pair to both members, once each."""
    rng = np.random.default_rng(22)
    y, pred = _labels(rng, 250, 5, 3)
    total = 0.0
    for a in range(5):
        for b in range(a + 1, 5):
            total += dv.double_fault(y, pred[:, a], pred[:, b])
    got = mdes.compute_pairwise_diversity(y, pred, mdes.DOUBLE_FAULT)
    assert got.sum() == pytest.approx(2.0 * total, rel=1e-12)
