"""Dynamic ensemble selection kernels for the deslib subset.

Ported from `deslib.util.diversity`, `deslib.util.aggregation`,
`deslib.util.prob_functions`, `deslib.util.knne` and
`deslib.des.des_clustering`.

Every exported symbol takes buffer addresses as plain `Int` and rebuilds the
pointer inside the body: `@export` rejects parametric functions, and an
inferred pointer origin would make the symbol parametric.
"""

from std.math import exp, fma, log, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]

# sys.float_info.max, which deslib returns for a degenerate ratio-of-errors.
comptime DBL_MAX = 1.7976931348623157e308


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


@export("dsl_diversity_matrix")
def dsl_diversity_matrix(
    y_addr: Int, pred_addr: Int, n: Int, m: Int, code: Int, out_addr: Int
) abi("C"):
    """Pairwise diversity matrix, matching `compute_pairwise_diversity`.

    `pred` is the (n, m) row-major class-label matrix produced by the pool,
    `y` is the (n,) target vector, and `out` is the (m,) accumulator that gets
    the pairwise value added to both members of each pair.

    The upstream implementation is three nested Python loops calling a Python
    function m*(m-1)/2 times, so the whole measure is dominated by interpreter
    overhead; here it collapses to one compiled sweep.

    `code` selects the measure:
        0 double fault, 1 negative double fault, 2 Q statistic,
        3 ratio of errors, 4 disagreement, 5 agreement, 6 correlation.
    """
    var y = ip(y_addr)
    var pred = ip(pred_addr)
    var out = fp(out_addr)

    for i in range(m):
        out[unsafe_offset=i] = 0.0

    var inv = 1.0 / Float64(n)
    for a in range(m):
        for b in range(a + 1, m):
            var n00 = 0.0
            var n10 = 0.0
            var n01 = 0.0
            var n11 = 0.0
            for s in range(n):
                var target = y[unsafe_offset=s]
                var ok_a = pred[unsafe_offset=s * m + a] == target
                var ok_b = pred[unsafe_offset=s * m + b] == target
                if ok_a:
                    if ok_b:
                        n11 += 1.0
                    else:
                        n10 += 1.0
                elif ok_b:
                    n01 += 1.0
                else:
                    n00 += 1.0
            n00 *= inv
            n10 *= inv
            n01 *= inv
            n11 *= inv

            var v = 0.0
            if code == 0:
                v = n00
            elif code == 1:
                v = -n00
            elif code == 2:
                v = ((n11 * n00) - (n01 * n10)) / ((n11 * n00) + (n01 * n10))
            elif code == 3:
                if n00 == 0.0:
                    v = DBL_MAX
                else:
                    v = (n01 + n10) / n00
            elif code == 4:
                v = n10 + n01
            elif code == 5:
                v = n00 + n11
            else:
                v = ((n11 * n00) - (n10 * n01)) / sqrt(
                    (n11 + n01) * (n10 + n00) * (n11 + n10) * (n01 + n00)
                )
            out[unsafe_offset=a] += v
            out[unsafe_offset=b] += v


@export("dsl_knn_kneighbors")
def dsl_knn_kneighbors(
    q_addr: Int,
    t_addr: Int,
    nq: Int,
    nt: Int,
    d: Int,
    k: Int,
    od_addr: Int,
    oi_addr: Int,
    row0: Int,
    row1: Int,
) abi("C"):
    """k nearest training rows under Euclidean distance, one row range at a time.

    This is the brute-force search behind the competence-region k-NN oracles
    (`KNNE` delegates to `KNeighborsClassifier(algorithm="brute")`). The
    squared distances are accumulated with FMA and the k smallest are kept in a
    sorted register file, so the selection is a streaming top-k with no
    allocation and no sorting of the full distance vector.

    Equal distances are broken by ascending training index, so the result is
    deterministic; upstream's `argpartition` gives no such guarantee.

    `row0`/`row1` let the shim split queries across worker threads.
    """
    var q = fp(q_addr)
    var t = fp(t_addr)
    var od = fp(od_addr)
    var oi = ip(oi_addr)

    for qi in range(row0, row1):
        var base_out = qi * k
        for j in range(k):
            od[unsafe_offset=base_out + j] = 0.0
            oi[unsafe_offset=base_out + j] = -1
        var filled = 0
        var worst = 0.0
        for ti in range(nt):
            var acc = 0.0
            var tbase = ti * d
            for f in range(d):
                var diff = q[unsafe_offset=qi * d + f] - t[unsafe_offset=tbase + f]
                acc = fma(diff, diff, acc)
            if filled < k:
                var pos = filled
                while pos > 0 and od[unsafe_offset=base_out + pos - 1] > acc:
                    od[unsafe_offset=base_out + pos] = od[
                        unsafe_offset=base_out + pos - 1
                    ]
                    oi[unsafe_offset=base_out + pos] = oi[
                        unsafe_offset=base_out + pos - 1
                    ]
                    pos -= 1
                od[unsafe_offset=base_out + pos] = acc
                oi[unsafe_offset=base_out + pos] = Int64(ti)
                filled += 1
                if filled == k:
                    worst = od[unsafe_offset=base_out + k - 1]
            elif acc < worst:
                var pos = k - 1
                while pos > 0 and od[unsafe_offset=base_out + pos - 1] > acc:
                    od[unsafe_offset=base_out + pos] = od[
                        unsafe_offset=base_out + pos - 1
                    ]
                    oi[unsafe_offset=base_out + pos] = oi[
                        unsafe_offset=base_out + pos - 1
                    ]
                    pos -= 1
                od[unsafe_offset=base_out + pos] = acc
                oi[unsafe_offset=base_out + pos] = Int64(ti)
                worst = od[unsafe_offset=base_out + k - 1]
        # The callers want true Euclidean distance, not its square; sqrt is
        # monotone so the ordering is unchanged.
        for j in range(k):
            od[unsafe_offset=base_out + j] = sqrt(od[unsafe_offset=base_out + j])


@export("dsl_sum_votes_per_class")
def dsl_sum_votes_per_class(
    pred_addr: Int, n: Int, m: Int, n_classes: Int, out_addr: Int
) abi("C"):
    """Vote histogram, matching `sum_votes_per_class`.

    `pred` is (n, m) integer class labels and `out` is (n, n_classes) counts.
    Labels outside [0, n_classes) are dropped, exactly as upstream's
    `np.sum(predictions == label, axis=1)` loop does. The output is integral,
    so the shim may compare it with exact equality.
    """
    var pred = ip(pred_addr)
    var out = ip(out_addr)
    for s in range(n):
        var base = s * n_classes
        for l in range(n_classes):
            out[unsafe_offset=base + l] = 0
        for j in range(m):
            var v = pred[unsafe_offset=s * m + j]
            if v >= 0 and v < Int64(n_classes):
                out[unsafe_offset=base + Int(v)] += 1


@export("dsl_reduce_rule")
def dsl_reduce_rule(
    pred_addr: Int, n: Int, m: Int, n_classes: Int, code: Int, scratch_addr: Int,
    out_addr: Int
) abi("C"):
    """Fuse the (n, m, n_classes) support tensor down the classifier axis.

    `code` selects the upstream rule: 0 average, 1 product, 2 median,
    3 maximum, 4 minimum. The caller then argmaxes the (n, n_classes) result,
    which is what every `*_rule` helper in `deslib.util.aggregation` does.

    `scratch` is a float64 row of length m, used only by the median rule, which
    needs a sorted copy. It is passed in so this kernel never allocates.
    """
    var pred = fp(pred_addr)
    var scratch = fp(scratch_addr)
    var out = fp(out_addr)
    var inv_m = 1.0 / Float64(m)

    for s in range(n):
        var sbase = s * m * n_classes
        for l in range(n_classes):
            var v = 0.0
            if code == 0:
                for j in range(m):
                    v += pred[unsafe_offset=sbase + j * n_classes + l]
                v *= inv_m
            elif code == 1:
                v = 1.0
                for j in range(m):
                    v *= pred[unsafe_offset=sbase + j * n_classes + l]
            elif code == 3:
                v = pred[unsafe_offset=sbase + l]
                for j in range(1, m):
                    var x = pred[unsafe_offset=sbase + j * n_classes + l]
                    if x > v:
                        v = x
            elif code == 4:
                v = pred[unsafe_offset=sbase + l]
                for j in range(1, m):
                    var x = pred[unsafe_offset=sbase + j * n_classes + l]
                    if x < v:
                        v = x
            else:
                for j in range(m):
                    scratch[unsafe_offset=j] = pred[
                        unsafe_offset=sbase + j * n_classes + l
                    ]
                for j in range(1, m):
                    var key = scratch[unsafe_offset=j]
                    var p = j
                    while p > 0 and scratch[unsafe_offset=p - 1] > key:
                        scratch[unsafe_offset=p] = scratch[unsafe_offset=p - 1]
                        p -= 1
                    scratch[unsafe_offset=p] = key
                if m % 2 == 1:
                    v = scratch[unsafe_offset=m // 2]
                else:
                    v = (
                        scratch[unsafe_offset=m // 2 - 1]
                        + scratch[unsafe_offset=m // 2]
                    ) * 0.5
            out[unsafe_offset=s * n_classes + l] = v


@export("dsl_argmax_rows")
def dsl_argmax_rows(a_addr: Int, rows: Int, cols: Int, out_addr: Int) abi("C"):
    """Row argmax with lowest-index tie-breaking, matching `np.argmax(axis=1)`.

    Indices, so the result is exact.
    """
    var a = fp(a_addr)
    var out = ip(out_addr)
    for s in range(rows):
        var base = s * cols
        var best = 0
        var bestv = a[unsafe_offset=base]
        for l in range(1, cols):
            var v = a[unsafe_offset=base + l]
            if v > bestv:
                bestv = v
                best = l
        out[unsafe_offset=s] = Int64(best)


@export("dsl_aggregate_weighted")
def dsl_aggregate_weighted(
    proba_addr: Int, w_addr: Int, n: Int, m: Int, n_classes: Int, out_addr: Int
) abi("C"):
    """`aggregate_proba_ensemble_weighted`: mean of proba*w over classifiers,
    then a row-wise softmax."""
    var proba = fp(proba_addr)
    var w = fp(w_addr)
    var out = fp(out_addr)
    var inv_m = 1.0 / Float64(m)

    for s in range(n):
        var sbase = s * m * n_classes
        var wbase = s * m
        var o = 0.0
        for l in range(n_classes):
            var acc = 0.0
            for j in range(m):
                acc = fma(
                    proba[unsafe_offset=sbase + j * n_classes + l],
                    w[unsafe_offset=wbase + j],
                    acc,
                )
            acc *= inv_m
            out[unsafe_offset=s * n_classes + l] = acc
            if acc > o:
                o = acc
        var total = 0.0
        for l in range(n_classes):
            var e = exp(out[unsafe_offset=s * n_classes + l] - o)
            out[unsafe_offset=s * n_classes + l] = e
            total += e
        for l in range(n_classes):
            out[unsafe_offset=s * n_classes + l] /= total


@export("dsl_softmax")
def dsl_softmax(w_addr: Int, rows: Int, cols: Int, theta: Float64, out_addr: Int) abi(
    "C"
):
    """`deslib.util.prob_functions.softmax`, shifted by the row max for
    stability."""
    var w = fp(w_addr)
    var out = fp(out_addr)
    for s in range(rows):
        var base = s * cols
        var o = w[unsafe_offset=base] / theta
        for l in range(1, cols):
            var v = w[unsafe_offset=base + l] / theta
            if v > o:
                o = v
        var total = 0.0
        for l in range(cols):
            var e = exp(w[unsafe_offset=base + l] / theta - o)
            out[unsafe_offset=base + l] = e
            total += e
        for l in range(cols):
            out[unsafe_offset=base + l] /= total


@export("dsl_exponential_competence")
def dsl_exponential_competence(
    n_classes: Int, sup_addr: Int, n: Int, out_addr: Int
) abi("C"):
    """`prob_functions.exponential_func`; saturates to 1 at full support."""
    var sup = fp(sup_addr)
    var out = fp(out_addr)
    var ln2 = log(2.0)
    var k = Float64(n_classes - 1)
    for i in range(n):
        var s = sup[unsafe_offset=i]
        if s <= 0.0:
            s = 0.0
        var v = 1.0
        if s < 1.0:
            v = 1.0 - exp(((1.0 - (k * s) / (1.0 - s)) * ln2))
        out[unsafe_offset=i] = v


@export("dsl_log_competence")
def dsl_log_competence(n_classes: Int, sup_addr: Int, n: Int, out_addr: Int) abi("C"):
    """`prob_functions.log_func`.

    `s ** temp` is evaluated as `exp(temp * log(s))` rather than through a
    `pow` intrinsic: Mojo's `pow` is a fast polynomial approximation and
    drifts from glibc's by ~2e-7 relative on these exponents, which is far
    larger than the difference a log and an exp introduce. `s == 0` falls out
    of `log(0) == -inf` and gives the same 0 the reference produces.
    """
    var sup = fp(sup_addr)
    var out = fp(out_addr)
    var temp = 1.0
    if n_classes != 2:
        temp = log(2.0) / log(Float64(n_classes))
    for i in range(n):
        var s = sup[unsafe_offset=i]
        if s > 1.0:
            s = 1.0
        elif s < 0.0:
            s = 0.0
        out[unsafe_offset=i] = 2.0 * exp(temp * log(s)) - 1.0


@export("dsl_entropy_competence")
def dsl_entropy_competence(
    sup_addr: Int, ok_addr: Int, n: Int, n_classes: Int, out_addr: Int
) abi("C"):
    """`prob_functions.entropy_func`.

    `scipy.stats.entropy` renormalises each row before taking the natural-log
    sum, so the kernel does the same; `ok` is the +1/-1 correctness flag.

    A support of exactly zero skips the term, matching `scipy.special.entr(0)`
    and the 0*log(0) convention. An all-zero row normalises to 0/0 and
    therefore yields nan here exactly as it does upstream.
    """
    var sup = fp(sup_addr)
    var ok = fp(ok_addr)
    var out = fp(out_addr)
    var inv = 1.0 / log(Float64(n_classes))
    for s in range(n):
        var base = s * n_classes
        var total = 0.0
        for l in range(n_classes):
            var v = sup[unsafe_offset=base + l]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            total += v
        var h = 0.0
        for l in range(n_classes):
            var v = sup[unsafe_offset=base + l]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            var p = v / total
            # entr(0) is 0; anything else (including nan) goes through log.
            if p != 0.0:
                h -= p * log(p)
        out[unsafe_offset=s] = h * inv + 2.0 * ok[unsafe_offset=s] - 1.0


@export("dsl_min_difference_competence")
def dsl_min_difference_competence(
    sup_addr: Int, idx_addr: Int, n: Int, n_classes: Int, out_addr: Int
) abi("C"):
    """`prob_functions.min_difference`: the smallest margin between the support
    for the correct class and the support for any other class."""
    var sup = fp(sup_addr)
    var idx = ip(idx_addr)
    var out = fp(out_addr)
    for s in range(n):
        var base = s * n_classes
        var c = idx[unsafe_offset=s]
        if c < 0 or c >= Int64(n_classes):
            out[unsafe_offset=s] = 0.0
            continue
        var correct = sup[unsafe_offset=base + Int(c)]
        var best = 0.0
        var seen = 0
        for l in range(n_classes):
            if Int64(l) == c:
                continue
            var diff = correct - sup[unsafe_offset=base + l]
            if seen == 0 or diff < best:
                best = diff
                seen = 1
        out[unsafe_offset=s] = best


@export("dsl_cluster_scores")
def dsl_cluster_scores(
    bks_addr: Int, target_addr: Int, idx_addr: Int, n_idx: Int, n_classes: Int,
    out_addr: Int
) abi("C"):
    """Per-classifier accuracy inside one cluster, the inner reduction of
    `DESClustering._preprocess_clusters`.

    `idx` lists the DSEL sample indices of one cluster and `bks` is the
    (n_samples, n_classifiers) pool prediction matrix; `out` receives one
    score per classifier.
    """
    var bks = ip(bks_addr)
    var target = ip(target_addr)
    var idx = i32p(idx_addr)
    var out = fp(out_addr)
    for l in range(n_classes):
        out[unsafe_offset=l] = 0.0
    if n_idx == 0:
        return
    for p in range(n_idx):
        var s = Int(idx[unsafe_offset=p])
        var t = target[unsafe_offset=s]
        for l in range(n_classes):
            if bks[unsafe_offset=s * n_classes + l] == t:
                out[unsafe_offset=l] += 1.0
    var inv = 1.0 / Float64(n_idx)
    for l in range(n_classes):
        out[unsafe_offset=l] *= inv
