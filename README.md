# mojo-deslib

A Mojo port of the compute core of [DESlib](https://github.com/scikit-learn-contrib/DESlib)
0.3.7, the dynamic ensemble selection library.

`deslib` is not a numeric library in the numpy-kernel sense: most of it is
scikit-learn plumbing. But it has three places where the work is a genuine
inner loop over sample-by-classifier arrays, and all three are on the hot path
of every DES/DCS method. Those are what is ported here. Everything else —
classifier fitting, the competence-region policies, the probabilistic DES
variants — stays in Python.

## What is ported

| Upstream symbol | Kind | Mojo kernel | Parity tested against |
| --- | --- | --- | --- |
| `util.diversity.compute_pairwise_diversity` and the seven measures it drives (`double_fault`, `negative_double_fault`, `Q_statistic`, `ratio_errors`, `disagreement_measure`, `agreement_measure`, `correlation_coefficient`) | values + index | `dsl_diversity_matrix` | `deslib.util.diversity`, exactly |
| `util.knne.KNNE` competence region (brute-force k-NN) | values + index | `dsl_knn_kneighbors` | `sklearn.neighbors.KNeighborsClassifier(algorithm="brute")` |
| `util.aggregation.sum_votes_per_class` | index | `dsl_sum_votes_per_class` | `deslib.util.aggregation`, exact integers |
| `util.aggregation.average_rule` / `product_rule` / `median_rule` / `maximum_rule` / `minimum_rule` | index | `dsl_reduce_rule` + `dsl_argmax_rows` | `deslib.util.aggregation`, exact indices |
| `util.aggregation.aggregate_proba_ensemble_weighted` | values | `dsl_aggregate_weighted` | `deslib.util.aggregation` |
| `util.prob_functions.softmax` | values | `dsl_softmax` | `deslib.util.prob_functions.softmax` |
| `util.prob_functions.exponential_func` | values | `dsl_exponential_competence` | `deslib.util.prob_functions` |
| `util.prob_functions.log_func` | values | `dsl_log_competence` | `deslib.util.prob_functions` |
| `util.prob_functions.entropy_func` | values | `dsl_entropy_competence` | `deslib.util.prob_functions` (+ `scipy.stats.entropy`) |
| `util.prob_functions.min_difference` | values | `dsl_min_difference_competence` | `deslib.util.prob_functions` |
| `des.des_clustering.DESClustering._preprocess_clusters` inner reduction | values | `dsl_cluster_scores` | `sklearn.metrics.accuracy_score` applied per pool member |

`DESlib` has no Fisher-score implementation in 0.3.7 — that lives in some
`deslibpy` forks, not in the released package. The nearest real equivalent,
the per-cluster per-classifier accuracy used by DES-Clustering, is ported.

## What is deliberately not ported

- `BaseDS`, `BaseDCS` and every `DES*` / `DCS*` estimator class. They are
  scikit-learn estimator boilerplate: fitting, validation, DFP pruning, region
  definition, selection policy. The numeric parts they call are here.
- `util.dfp` (frienemy pruning) and `util.instance_hardness` — both reduce
  over a handful of scalars after the neighbour search, so a compiled kernel
  would be slower than the NumPy they already use.
- `util.diversity_batch` — a batch API with the same arithmetic as
  `util.diversity`.
- `util.datasets` — file IO.
- `util.faiss_knn_wrapper` — a thin optional delegate to FAISS.
- `util.prob_functions.ccprmod` — it needs the incomplete beta function over a
  `B x C` grid per sample; that belongs to SciPy's special functions, not to a
  Mojo kernel, and reimplementing `betainc` would be a port of SciPy, not of
  DESlib.
- The `probabilistic` DES family (DESKL, RRC, META-DES) — model fitting in
  scikit-learn.

## Behavioural differences

Two, both tested:

1. `Q_statistic` on a pair where `N11*N00 + N01*N10 == 0` raises
   `ZeroDivisionError` in deslib (it divides Python floats). The kernel has no
   exception path across the C ABI and returns `nan` instead.
2. `log_competence` evaluates `s ** temp` as `exp(temp * log(s))`, and
   `entropy_competence` uses Mojo's `log`/`exp`. On this toolchain those
   intrinsics are accurate to about `1e-9` relative where glibc's are good to
   about `1e-16`, so the two transcendental competence measures are compared
   at `rtol=1e-7, atol=1e-8`. Every non-transcendental measure is compared at
   `1e-12` or exactly, as the table above indicates.

The k-NN kernel breaks equal distances by ascending training index.
scikit-learn's `argpartition`-based selection gives no such guarantee, but on
continuous data ties do not occur, so the parity test uses continuous data and
a separate test pins the tie rule.

## Install and build

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh                    # -> dist/libmojo-deslib.so
PYTHONPATH=python python -m pytest tests -q
```

The Python package is `mojo_deslib` and imports alongside the real `deslib`;
it never imports `deslib` itself.

## Benchmark

`python bench/bench.py` checks every case against the reference before timing
it. Measured on this box, Mojo 1.2.0.dev2026092605, Xeon E5-2697 v4, best of
3-5 runs:

| case | reference | mojo | speedup |
| --- | ---: | ---: | ---: |
| diversity n=4000 pool=24 | 2629.666ms | 12.539ms | 209.72x |
| kneighbors q=4096 train=20000 d=32 k=7 [8w] | 4704.479ms | 2200.723ms | 2.14x |
| average_rule n=20000 pool=12 classes=6 | 15.453ms | 2.580ms | 5.99x |
| median_rule n=20000 pool=12 classes=6 | 139.267ms | 56.818ms | 2.45x |
| softmax n=524288 cols=16 | 444.255ms | 278.609ms | 1.59x |
| weighted_aggregate n=20000 pool=10 classes=5 | 24.256ms | 3.650ms | 6.65x |
| sum_votes n=1048576 pool=10 classes=4 | 275.020ms | 64.929ms | 4.24x |
| entropy_competence n=262144 classes=8 | 167.069ms | 62.050ms | 2.69x |
| cluster_scores n=65536 pool=30 | 118.755ms | 5.160ms | 23.01x |

Reading the numbers:

- **diversity, 210x.** The reference is three nested Python loops that call a
  Python function `m*(m-1)/2` times. There is no vectorised NumPy equivalent
  in the package, and this is exactly the case the port exists for.
- **cluster_scores, 23x.** The reference is
  `np.apply_along_axis(precision_function, 0, ...)` with a Python callable per
  column.
- **kneighbors, 2.14x with 8 threads and 0.56x without.** The kernel is
  compute-bound (`d = 32` flops per distance) so the shim splits the query
  rows over a thread pool; Mojo 1.2.0 cannot pass a pointer into a
  `parallelize` body, so the split happens in Python and ctypes releases the
  GIL for the foreign call. Serially the streaming top-k loses to sklearn's
  blocked, BLAS-backed distance computation — that is reported, not hidden.
- **softmax, 1.59x.** Bandwidth-bound, so this is close to parity with a
  single-threaded NumPy expression chain; the win is one pass instead of
  three.

The box is shared, so absolute times move by tens of percent between runs; the
parity assertions in `bench/bench.py` are the part that does not move.
