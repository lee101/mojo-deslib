"""mojo-deslib: dynamic ensemble selection with Mojo kernels.

Installable alongside the real `deslib` package, which it is tested against
for parity.
"""

from ._lib import (
    aggregate_proba_ensemble_weighted,
    argmax_rows,
    cluster_scores,
    compute_pairwise_diversity,
    entropy_competence,
    exponential_competence,
    fuse_rule,
    kneighbors,
    log_competence,
    min_difference_competence,
    softmax,
    sum_votes_per_class,
)

DOUBLE_FAULT = 0
NEGATIVE_DOUBLE_FAULT = 1
Q_STATISTIC = 2
RATIO_ERRORS = 3
DISAGREEMENT = 4
AGREEMENT = 5
CORRELATION = 6

__all__ = [
    "aggregate_proba_ensemble_weighted",
    "argmax_rows",
    "cluster_scores",
    "compute_pairwise_diversity",
    "entropy_competence",
    "exponential_competence",
    "fuse_rule",
    "kneighbors",
    "log_competence",
    "min_difference_competence",
    "softmax",
    "sum_votes_per_class",
    "DOUBLE_FAULT",
    "NEGATIVE_DOUBLE_FAULT",
    "Q_STATISTIC",
    "RATIO_ERRORS",
    "DISAGREEMENT",
    "AGREEMENT",
    "CORRELATION",
]
__version__ = "0.1.0"
