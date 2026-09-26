"""Parity of the aggregation and competence kernels against `deslib`."""

import numpy as np
import pytest

import mojo_deslib as mdes

from deslib.util import aggregation as agg
from deslib.util import prob_functions as pf

RULES = [
    ("average", agg.average_rule),
    ("product", agg.product_rule),
    ("median", agg.median_rule),
    ("maximum", agg.maximum_rule),
    ("minimum", agg.minimum_rule),
]


# Mojo's exp/log on this build are accurate to ~1e-9 relative, against
# glibc's ~1e-16, and every competence measure here is a difference of
# transcendental terms that crosses zero. A relative comparison is therefore
# meaningless on part of the range; 1e-7/1e-8 is the honest floor and is
# still four orders tighter than any decision these measures drive.
RTOL = 1e-7
ATOL = 1e-8


def _assert_close(got, want):
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def _proba(rng, n, m, c):
    p = rng.random((n, m, c))
    return p / p.sum(axis=2, keepdims=True)


@pytest.mark.parametrize("rule,func", RULES, ids=[r for r, _ in RULES])
@pytest.mark.parametrize("m", [1, 2, 5, 8])
def test_fuse_rule_matches_deslib(rule, func, m):
    """Odd and even pool sizes both matter: even m takes the midpoint median."""
    rng = np.random.default_rng(50 + m)
    p = _proba(rng, 300, m, 4)
    got = mdes.fuse_rule(p, rule)
    want = np.asarray(func(p))
    np.testing.assert_array_equal(got, want)


def test_fuse_rule_ties_go_to_lowest_class_index():
    """np.argmax breaks ties toward 0; the kernel must do the same or a tied
    row would pick a different class and a downstream accuracy check would
    silently change."""
    p = np.zeros((3, 2, 3))
    p[:, :, 0] = 0.5
    p[:, :, 1] = 0.5
    p[:, :, 2] = 0.0
    np.testing.assert_array_equal(mdes.fuse_rule(p, "average"), [0, 0, 0])


def test_sum_votes_per_class_is_exact():
    rng = np.random.default_rng(61)
    pred = rng.integers(0, 5, size=(400, 7))
    got = mdes.sum_votes_per_class(pred, 5)
    want = agg.sum_votes_per_class(pred, 5)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got.sum(axis=1), np.full(400, 7))


def test_sum_votes_per_class_drops_out_of_range_labels():
    """Upstream only counts labels in [0, n_classes); a kernel that indexed
    blindly would corrupt the last column."""
    pred = np.array([[0, 1, 9], [2, 2, -1]], dtype=np.int64)
    got = mdes.sum_votes_per_class(pred, 3)
    np.testing.assert_array_equal(got, agg.sum_votes_per_class(pred, 3))
    np.testing.assert_array_equal(got, [[1, 1, 0], [0, 0, 2]])


def test_argmax_rows_exact():
    a = np.array([[1.0, 3.0, 3.0], [-1.0, -1.0, -2.0]])
    np.testing.assert_array_equal(mdes.argmax_rows(a), [1, 0])


def test_aggregate_proba_ensemble_weighted_matches_deslib():
    rng = np.random.default_rng(71)
    n, m, c = 250, 6, 3
    proba = _proba(rng, n, m, c)
    w = rng.random((n, m))
    got = mdes.aggregate_proba_ensemble_weighted(proba, w)
    want = agg.aggregate_proba_ensemble_weighted(proba, w)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-12)


def test_aggregate_weighted_uniform_weights_reduce_to_average():
    rng = np.random.default_rng(72)
    n, m, c = 40, 5, 4
    proba = _proba(rng, n, m, c)
    w = np.ones((n, m))
    got = mdes.aggregate_proba_ensemble_weighted(proba, w)
    mean = proba.mean(axis=1)
    np.testing.assert_allclose(got, pf.softmax(mean), rtol=1e-12, atol=1e-15)


def test_softmax_matches_deslib_and_is_stable():
    rng = np.random.default_rng(73)
    a = rng.standard_normal((50, 7)) * 40.0
    got = mdes.softmax(a)
    want = pf.softmax(a)
    assert np.all(np.isfinite(got)), "large logits overflowed"
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-12)


def test_softmax_respects_theta():
    a = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        mdes.softmax(a, 10.0), pf.softmax(a, 10.0), rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("n_classes", [2, 3, 6])
def test_exponential_competence_matches_deslib(n_classes):
    rng = np.random.default_rng(80 + n_classes)
    s = rng.random(500)
    got = mdes.exponential_competence(n_classes, s)
    want = pf.exponential_func(n_classes, s.copy())
    _assert_close(got, want)


def test_exponential_competence_saturates_at_one():
    """The support-one branch must short-circuit: the formula below divides by
    `1 - support`, so evaluating it at 1 would produce -inf or a nan instead
    of the documented 1."""
    s = np.array([0.0, 1.0, 1.5, -0.2])
    got = mdes.exponential_competence(3, s)
    np.testing.assert_allclose(
        got, pf.exponential_func(3, s.copy()), rtol=1e-12, atol=1e-13
    )
    assert got[1] == 1.0 and got[2] == 1.0
    # Zero support gives 1 - 2**1, which is -1, not 0.
    assert got[0] == pytest.approx(-1.0, abs=1e-12)




@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_log_competence_matches_deslib(n_classes):
    rng = np.random.default_rng(90 + n_classes)
    s = rng.random(400)
    got = mdes.log_competence(n_classes, s)
    want = pf.log_func(n_classes, s.copy())
    _assert_close(got, want)


def test_log_competence_clamps_out_of_range_support():
    got = mdes.log_competence(3, np.array([-1.0, 0.0, 1.0, 2.0]))
    want = pf.log_func(3, np.array([-1.0, 0.0, 1.0, 2.0]))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert got[0] == -1.0 and got[2] == 1.0


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_entropy_competence_matches_deslib(n_classes):
    rng = np.random.default_rng(100 + n_classes)
    s = rng.random((300, n_classes))
    s /= s.sum(axis=1, keepdims=True)
    ok = rng.choice([-1.0, 1.0], size=300)
    got = mdes.entropy_competence(n_classes, s, ok)
    want = pf.entropy_func(n_classes, s.copy(), ok.copy())
    _assert_close(got, want)


def test_entropy_competence_sign_follows_correctness():
    """A uniform support has maximal entropy, `log(n_classes)`, so the entropy
    term is exactly 1 and only the +-1 correctness flag remains."""
    n_classes = 4
    s = np.full((2, n_classes), 0.25)
    ok = np.array([1.0, -1.0])
    got = mdes.entropy_competence(n_classes, s, ok)
    _assert_close(got, np.array([2.0, -2.0]))


def test_entropy_competence_handles_zero_support():
    """scipy's entr(0) is 0, so a zero column must contribute nothing."""
    n_classes = 3
    s = np.array([[1.0, 0.0, 0.0], [0.4, 0.6, 0.0]])
    ok = np.array([1.0, -1.0])
    got = mdes.entropy_competence(n_classes, s, ok)
    want = pf.entropy_func(n_classes, s.copy(), ok.copy())
    _assert_close(got, want)
    # A one-hot row has zero entropy, so the competence is just 2*ok - 1.
    assert got[0] == pytest.approx(1.0, abs=1e-12)


def test_entropy_competence_all_zero_row_is_nan_like_scipy():
    """Documented divergence-free behaviour: a row that cannot be normalised
    gives nan, matching the 0/0 scipy performs in the same place."""
    n_classes = 3
    s = np.zeros((1, n_classes))
    got = mdes.entropy_competence(n_classes, s, np.array([1.0]))
    want = pf.entropy_func(n_classes, s.copy(), np.array([1.0]))
    assert np.isnan(got[0]) and np.isnan(want[0])


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_min_difference_matches_deslib(n_classes):
    rng = np.random.default_rng(110 + n_classes)
    s = rng.random((200, n_classes))
    s /= s.sum(axis=1, keepdims=True)
    idx = rng.integers(0, n_classes, size=200)
    got = mdes.min_difference_competence(s, idx)
    want = pf.min_difference(s.copy(), idx)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_min_difference_is_negative_when_misclassified():
    s = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    got = mdes.min_difference_competence(s, np.array([0, 0]))
    assert got[0] < 0.0
    assert got[1] > 0.0


def test_cluster_scores_matches_accuracy():
    """`DESClustering._preprocess_clusters` scores each pool member by its
    accuracy over the cluster's samples."""
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(120)
    n, m, c = 500, 7, 3
    bks = rng.integers(0, c, size=(n, m))
    target = rng.integers(0, c, size=n)
    idx = np.sort(rng.choice(n, size=120, replace=False))
    got = mdes.cluster_scores(bks, target, idx)
    want = np.array([accuracy_score(target[idx], bks[idx, j]) for j in range(m)])
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_cluster_scores_empty_cluster_is_all_zero():
    bks = np.zeros((10, 3), dtype=np.int64)
    target = np.zeros(10, dtype=np.int64)
    got = mdes.cluster_scores(bks, target, np.array([], dtype=np.int32))
    np.testing.assert_array_equal(got, np.zeros(3))
