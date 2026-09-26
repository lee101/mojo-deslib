"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the address argtypes must stay `c_int64`; `c_int` truncates them and
segfaults.
"""

import ctypes
import os
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-deslib.so"

_i64 = ctypes.c_int64
_f64 = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.dsl_diversity_matrix.restype = None
    lib.dsl_diversity_matrix.argtypes = [_i64] * 6

    lib.dsl_knn_kneighbors.restype = None
    lib.dsl_knn_kneighbors.argtypes = [_i64] * 10

    lib.dsl_sum_votes_per_class.restype = None
    lib.dsl_sum_votes_per_class.argtypes = [_i64] * 5

    lib.dsl_reduce_rule.restype = None
    lib.dsl_reduce_rule.argtypes = [_i64] * 7

    lib.dsl_argmax_rows.restype = None
    lib.dsl_argmax_rows.argtypes = [_i64] * 4

    lib.dsl_aggregate_weighted.restype = None
    lib.dsl_aggregate_weighted.argtypes = [_i64] * 6

    lib.dsl_softmax.restype = None
    lib.dsl_softmax.argtypes = [_i64, _i64, _i64, _f64, _i64]

    lib.dsl_exponential_competence.restype = None
    lib.dsl_exponential_competence.argtypes = [_i64, _i64, _i64, _i64]

    lib.dsl_log_competence.restype = None
    lib.dsl_log_competence.argtypes = [_i64, _i64, _i64, _i64]

    lib.dsl_entropy_competence.restype = None
    lib.dsl_entropy_competence.argtypes = [_i64] * 5

    lib.dsl_min_difference_competence.restype = None
    lib.dsl_min_difference_competence.argtypes = [_i64] * 5

    lib.dsl_cluster_scores.restype = None
    lib.dsl_cluster_scores.argtypes = [_i64] * 6
    return lib


lib = _load()




def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


# ---------------------------------------------------------------- diversity


def compute_pairwise_diversity(targets, prediction_matrix, code: int) -> np.ndarray:
    """`deslib.util.diversity.compute_pairwise_diversity` in one call.

    `code` is the measure selector: 0 double fault, 1 negative double fault,
    2 Q statistic, 3 ratio of errors, 4 disagreement, 5 agreement,
    6 correlation coefficient.
    """
    y = np.ascontiguousarray(targets, dtype=np.int64).reshape(-1)
    pred = np.ascontiguousarray(prediction_matrix, dtype=np.int64)
    if pred.ndim != 2:
        raise ValueError("prediction_matrix must be 2-D")
    n, m = pred.shape
    if y.shape[0] != n:
        raise ValueError("targets and prediction_matrix disagree on n_samples")
    out = np.empty(m, dtype=np.float64)
    lib.dsl_diversity_matrix(
        _addr(y), _addr(pred), n, m, int(code), _addr(out)
    )
    return out


# --------------------------------------------------------------------- kNN


def _chunk_bounds(nq: int, workers: int):
    if workers <= 1 or nq < 2 * workers:
        return [(0, nq)]
    step = (nq + workers - 1) // workers
    return [(i * step, min((i + 1) * step, nq)) for i in range(workers)
            if i * step < nq]


def kneighbors(query, train, k: int, workers: int = 1):
    """Brute-force Euclidean k nearest neighbours.

    Returns `(distances, indices)` sorted by increasing distance, matching
    `KNeighborsClassifier(n_neighbors=k, algorithm="brute").kneighbors`.
    """
    from concurrent.futures import ThreadPoolExecutor

    q = np.ascontiguousarray(query, dtype=np.float64)
    t = np.ascontiguousarray(train, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    nq, d = q.shape
    nt = t.shape[0]
    if t.shape[1] != d:
        raise ValueError("query and train have different feature counts")
    if k < 1 or k > nt:
        raise ValueError(f"k must be in [1, {nt}], got {k}")

    od = np.empty((nq, k), dtype=np.float64)
    oi = np.empty((nq, k), dtype=np.int64)
    parts = _chunk_bounds(nq, workers)
    if len(parts) == 1:
        lib.dsl_knn_kneighbors(
            _addr(q), _addr(t), nq, nt, d, k, _addr(od), _addr(oi), 0, nq
        )
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as ex:
            list(ex.map(
                lambda p: lib.dsl_knn_kneighbors(
                    _addr(q), _addr(t), nq, nt, d, k, _addr(od), _addr(oi),
                    p[0], p[1],
                ),
                parts,
            ))
    return od, oi


# ------------------------------------------------------------- aggregation


def sum_votes_per_class(predictions, n_classes: int) -> np.ndarray:
    pred = np.ascontiguousarray(predictions, dtype=np.int64)
    if pred.ndim != 2:
        raise ValueError("predictions must be 2-D")
    n, m = pred.shape
    out = np.empty((n, n_classes), dtype=np.int64)
    lib.dsl_sum_votes_per_class(
        _addr(pred), n, m, int(n_classes), _addr(out)
    )
    return out


_RULE_CODES = {
    "average": 0,
    "product": 1,
    "median": 2,
    "maximum": 3,
    "minimum": 4,
}


def fuse_rule(predictions, rule: str) -> np.ndarray:
    """Apply one of the `deslib.util.aggregation` fusion rules.

    Returns the per-sample argmax class index, which is what the upstream
    `*_rule` helpers return.
    """
    pred = np.ascontiguousarray(predictions, dtype=np.float64)
    if pred.ndim != 3:
        raise ValueError("predictions must be (n_samples, n_classifiers, n_classes)")
    n, m, c = pred.shape
    scratch = np.empty(m, dtype=np.float64)
    fused = np.empty((n, c), dtype=np.float64)
    lib.dsl_reduce_rule(
        _addr(pred), n, m, c, _RULE_CODES[rule], _addr(scratch), _addr(fused)
    )
    labels = np.empty(n, dtype=np.int64)
    lib.dsl_argmax_rows(_addr(fused), n, c, _addr(labels))
    return labels


def argmax_rows(values) -> np.ndarray:
    a = np.ascontiguousarray(values, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("values must be 2-D")
    rows, cols = a.shape
    out = np.empty(rows, dtype=np.int64)
    lib.dsl_argmax_rows(_addr(a), rows, cols, _addr(out))
    return out


def aggregate_proba_ensemble_weighted(ensemble_proba, weights) -> np.ndarray:
    """`aggregate_proba_ensemble_weighted`: mean of proba*w then softmax."""
    proba = np.ascontiguousarray(ensemble_proba, dtype=np.float64)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if proba.ndim != 3:
        raise ValueError("ensemble_proba must be 3-D")
    n, m, c = proba.shape
    if w.shape != (n, m):
        raise ValueError("weights must be (n_samples, n_classifiers)")
    out = np.empty((n, c), dtype=np.float64)
    lib.dsl_aggregate_weighted(
        _addr(proba), _addr(w), n, m, c, _addr(out)
    )
    return out


def softmax(w, theta: float = 1.0) -> np.ndarray:
    """`deslib.util.prob_functions.softmax`, row-wise."""
    a = np.atleast_2d(np.ascontiguousarray(w, dtype=np.float64))
    rows, cols = a.shape
    out = np.empty_like(a)
    lib.dsl_softmax(_addr(a), rows, cols, ctypes.c_double(theta), _addr(out))
    return out


# -------------------------------------------------------- competence scores


def exponential_competence(n_classes: int, support_correct) -> np.ndarray:
    sup = np.ascontiguousarray(support_correct, dtype=np.float64).reshape(-1)
    out = np.empty(sup.size, dtype=np.float64)
    lib.dsl_exponential_competence(
        int(n_classes), _addr(sup), sup.size, _addr(out)
    )
    return out


def log_competence(n_classes: int, support_correct) -> np.ndarray:
    sup = np.ascontiguousarray(support_correct, dtype=np.float64).reshape(-1)
    out = np.empty(sup.size, dtype=np.float64)
    lib.dsl_log_competence(int(n_classes), _addr(sup), sup.size, _addr(out))
    return out


def entropy_competence(n_classes: int, supports, is_correct) -> np.ndarray:
    sup = np.ascontiguousarray(supports, dtype=np.float64)
    ok = np.ascontiguousarray(is_correct, dtype=np.float64).reshape(-1)
    n, c = sup.shape
    out = np.empty(n, dtype=np.float64)
    lib.dsl_entropy_competence(
        _addr(sup), _addr(ok), n, int(n_classes), _addr(out)
    )
    return out


def min_difference_competence(supports, idx_correct_label) -> np.ndarray:
    sup = np.ascontiguousarray(supports, dtype=np.float64)
    idx = np.ascontiguousarray(idx_correct_label, dtype=np.int64).reshape(-1)
    n, c = sup.shape
    out = np.empty(n, dtype=np.float64)
    lib.dsl_min_difference_competence(
        _addr(sup), _addr(idx), n, c, _addr(out)
    )
    return out


# ------------------------------------------------------ DES-Clustering core


def cluster_scores(bks, targets, sample_indices) -> np.ndarray:
    """Per-classifier accuracy over the samples of one cluster.

    This is the reduction `DESClustering._preprocess_clusters` performs through
    `np.apply_along_axis(precision_function, 0, ...)` with the default accuracy
    metric.
    """
    bks = np.ascontiguousarray(bks, dtype=np.int64)
    tgt = np.ascontiguousarray(targets, dtype=np.int64).reshape(-1)
    idx = np.ascontiguousarray(sample_indices, dtype=np.int32).reshape(-1)
    _, m = bks.shape
    out = np.empty(m, dtype=np.float64)
    lib.dsl_cluster_scores(
        _addr(bks), _addr(tgt), _addr(idx), idx.size, m, _addr(out)
    )
    return out
