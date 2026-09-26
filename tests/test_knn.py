"""Parity of the brute-force k-NN search against scikit-learn.

`KNNE` in deslib delegates its competence region to
`KNeighborsClassifier(algorithm="brute")`, so sklearn is the reference.
"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import mojo_deslib as mdes


@pytest.mark.parametrize("k", [1, 3, 7, 15])
def test_kneighbors_matches_sklearn(k):
    rng = np.random.default_rng(20 + k)
    train = rng.standard_normal((600, 9))
    query = rng.standard_normal((120, 9))
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(
        train, np.zeros(len(train))
    )
    d_want, i_want = knn.kneighbors(query, return_distance=True)
    d_got, i_got = mdes.kneighbors(query, train, k)
    np.testing.assert_array_equal(i_got, i_want)
    np.testing.assert_allclose(d_got, d_want, rtol=1e-12, atol=1e-12)


def test_kneighbors_threaded_equals_serial():
    """The row-range split must not change the answer, only the timing."""
    rng = np.random.default_rng(404)
    train = rng.standard_normal((800, 12))
    query = rng.standard_normal((257, 12))
    d_serial, i_serial = mdes.kneighbors(query, train, 9, workers=1)
    d_par, i_par = mdes.kneighbors(query, train, 9, workers=8)
    np.testing.assert_array_equal(i_par, i_serial)
    np.testing.assert_allclose(d_par, d_serial, rtol=0, atol=0)


def test_kneighbors_breaks_ties_by_ascending_index():
    """Query [0,1] is equidistant from train rows 0 and 1.

    The documented rule is sort by (distance, index), so the answer is 0 then
    1. A kernel that appended on `dist <= worst` would emit them the other way
    round, and one that used `<` for the neighbour comparison would drop a
    true neighbour entirely.
    """
    train = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64
    )
    d, i = mdes.kneighbors(train, train, 3)
    np.testing.assert_array_equal(
        i, [[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 1, 2]]
    )
    np.testing.assert_allclose(
        d,
        np.sqrt([[0.0, 1.0, 1.0],
                 [0.0, 1.0, 2.0],
                 [0.0, 1.0, 2.0],
                 [0.0, 5.0, 5.0]]),
        rtol=0, atol=0,
    )


def test_kneighbors_distances_are_sorted():
    rng = np.random.default_rng(9)
    train = rng.standard_normal((300, 5))
    query = rng.standard_normal((50, 5))
    d, i = mdes.kneighbors(query, train, 10)
    assert np.all(np.diff(d, axis=1) >= -1e-12)
    # And every reported index really has that distance.
    for r in range(5):
        np.testing.assert_allclose(
            d[r], np.linalg.norm(train[i[r]] - query[r], axis=1),
            rtol=1e-12, atol=1e-12,
        )


def test_kneighbors_feature_count_is_actually_used():
    """A kernel that ignored `d` would return neighbours on the prefix only."""
    rng = np.random.default_rng(31)
    train = np.zeros((40, 4))
    train[:, 0] = np.arange(40)
    query = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    d, i = mdes.kneighbors(query, train, 2)
    assert i[0, 0] == 0
    assert i[1, 0] == 3
    np.testing.assert_allclose(d[0], [0.0, 1.0], rtol=0, atol=0)
    np.testing.assert_allclose(d[1], [0.0, 1.0], rtol=0, atol=0)


def test_kneighbors_rejects_bad_k():
    train = np.zeros((3, 2))
    with pytest.raises(ValueError):
        mdes.kneighbors(train, train, 0)
    with pytest.raises(ValueError):
        mdes.kneighbors(train, train, 4)
