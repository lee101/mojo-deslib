"""Correctness-gated benchmark for mojo-deslib.

Every case checks numerical agreement with the reference before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_deslib as mdes  # noqa: E402

from deslib.util import aggregation as agg  # noqa: E402
from deslib.util import diversity as dv  # noqa: E402
from deslib.util import prob_functions as pf  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_diversity(n: int = 4000, m: int = 24):
    """Pairwise diversity over a wide pool.

    The reference is deslib's own `compute_pairwise_diversity`, which is three
    nested Python loops; there is no vectorised NumPy equivalent to compare
    against, so this measures the interpreter overhead the port removes.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=n)
    pred = rng.integers(0, 4, size=(n, m))

    got = mdes.compute_pairwise_diversity(y, pred, mdes.Q_STATISTIC)
    want = dv.compute_pairwise_diversity(y, pred, dv.Q_statistic)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    ref = _time(lambda: dv.compute_pairwise_diversity(y, pred, dv.Q_statistic), 3)
    mojo = _time(
        lambda: mdes.compute_pairwise_diversity(y, pred, mdes.Q_STATISTIC), 3
    )
    return f"diversity n={n} pool={m}", ref, mojo


def bench_kneighbors(nq: int = 4096, nt: int = 20000, d: int = 32, k: int = 7,
                    workers: int = 8):
    """Brute-force k-NN against scikit-learn's brute-force neighbour search,
    which is the closest fair NumPy-flavoured baseline available here."""
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(1)
    train = rng.standard_normal((nt, d))
    query = rng.standard_normal((nq, d))
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(
        train, np.zeros(nt)
    )
    i_want = knn.kneighbors(query, return_distance=True)[1]
    d_got, i_got = mdes.kneighbors(query, train, k)
    np.testing.assert_array_equal(i_got, i_want)

    ref = _time(lambda: knn.kneighbors(query, return_distance=True), 3)
    mojo = _time(lambda: mdes.kneighbors(query, train, k, workers=workers), 3)
    return (f"kneighbors q={nq} train={nt} d={d} k={k} [{workers}w]",
            ref, mojo)


def bench_fuse_rule(n: int = 20000, m: int = 12, c: int = 6):
    """Average fusion over a large support tensor, against the vectorised NumPy
    formulation the upstream rule already uses."""
    rng = np.random.default_rng(2)
    p = rng.random((n, m, c))
    p /= p.sum(axis=2, keepdims=True)
    got = mdes.fuse_rule(p, "average")
    want = agg.average_rule(p)
    np.testing.assert_array_equal(got, want)

    ref = _time(lambda: agg.average_rule(p), 5)
    mojo = _time(lambda: mdes.fuse_rule(p, "average"), 5)
    return f"average_rule n={n} pool={m} classes={c}", ref, mojo


def bench_median_rule(n: int = 20000, m: int = 12, c: int = 6):
    """Median fusion needs a sort per cell; NumPy's partition is a strong
    baseline here, so a slowdown would be a real result."""
    rng = np.random.default_rng(3)
    p = rng.random((n, m, c))
    got = mdes.fuse_rule(p, "median")
    want = agg.median_rule(p)
    np.testing.assert_array_equal(got, want)

    ref = _time(lambda: agg.median_rule(p), 3)
    mojo = _time(lambda: mdes.fuse_rule(p, "median"), 3)
    return f"median_rule n={n} pool={m} classes={c}", ref, mojo


def bench_softmax(n: int = 1 << 19, cols: int = 16):
    rng = np.random.default_rng(4)
    w = rng.standard_normal((n, cols))
    np.testing.assert_allclose(
        mdes.softmax(w), pf.softmax(w), rtol=1e-9, atol=1e-12
    )
    ref = _time(lambda: pf.softmax(w), 5)
    mojo = _time(lambda: mdes.softmax(w), 5)
    return f"softmax n={n} cols={cols}", ref, mojo


def bench_aggregate_weighted(n: int = 20000, m: int = 10, c: int = 5):
    rng = np.random.default_rng(5)
    proba = rng.random((n, m, c))
    w = rng.random((n, m))
    got = mdes.aggregate_proba_ensemble_weighted(proba, w)
    want = agg.aggregate_proba_ensemble_weighted(proba, w)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    ref = _time(lambda: agg.aggregate_proba_ensemble_weighted(proba, w), 3)
    mojo = _time(lambda: mdes.aggregate_proba_ensemble_weighted(proba, w), 3)
    return f"weighted_aggregate n={n} pool={m} classes={c}", ref, mojo


def bench_sum_votes(n: int = 1 << 20, m: int = 10, c: int = 4):
    rng = np.random.default_rng(6)
    pred = rng.integers(0, c, size=(n, m))
    got = mdes.sum_votes_per_class(pred, c)
    want = agg.sum_votes_per_class(pred, c)
    np.testing.assert_array_equal(got, want)

    ref = _time(lambda: agg.sum_votes_per_class(pred, c), 3)
    mojo = _time(lambda: mdes.sum_votes_per_class(pred, c), 3)
    return f"sum_votes n={n} pool={m} classes={c}", ref, mojo


def bench_entropy_competence(n: int = 1 << 18, c: int = 8):
    rng = np.random.default_rng(7)
    s = rng.random((n, c))
    s /= s.sum(axis=1, keepdims=True)
    ok = rng.choice([-1.0, 1.0], size=n)
    got = mdes.entropy_competence(c, s, ok)
    want = pf.entropy_func(c, s.copy(), ok.copy())
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-8)

    ref = _time(lambda: pf.entropy_func(c, s.copy(), ok.copy()), 3)
    mojo = _time(lambda: mdes.entropy_competence(c, s, ok), 3)
    return f"entropy_competence n={n} classes={c}", ref, mojo


def bench_cluster_scores(n: int = 1 << 16, m: int = 30, frac: float = 0.25):
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(8)
    bks = rng.integers(0, 3, size=(n, m))
    target = rng.integers(0, 3, size=n)
    idx = np.sort(rng.choice(n, size=int(n * frac), replace=False))
    got = mdes.cluster_scores(bks, target, idx)
    want = np.array([accuracy_score(target[idx], bks[idx, j]) for j in range(m)])
    np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def ref():
        return np.apply_along_axis(
            lambda p: accuracy_score(target[idx], p), 0, bks[idx, :]
        )

    return f"cluster_scores n={n} pool={m}", _time(ref, 3), _time(
        lambda: mdes.cluster_scores(bks, target, idx), 3
    )


CASES = [
    bench_diversity,
    bench_kneighbors,
    bench_fuse_rule,
    bench_median_rule,
    bench_softmax,
    bench_aggregate_weighted,
    bench_sum_votes,
    bench_entropy_competence,
    bench_cluster_scores,
]


def main():
    print(f"{'case':<48} {'reference':>12} {'mojo':>12} {'speedup':>9}")
    print("-" * 85)
    for case in CASES:
        label, ref, mojo = case()
        print(f"{label:<48} {ref * 1e3:>10.3f}ms {mojo * 1e3:>10.3f}ms "
              f"{ref / mojo:>8.2f}x")


if __name__ == "__main__":
    main()
