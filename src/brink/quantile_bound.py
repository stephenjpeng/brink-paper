"""Feasible-fraction threshold U_eps from independently sampled designs (Prop. 1).

Let s : X -> [-inf, +inf] be a per-design upper limit with F(c) a subset of
{x : s(x) >= c}, V(c) = mu(F(c)), and theta_eps = sup{c : V(c) > eps} (sup of
the empty set = -inf). Draw X_1, ..., X_N i.i.d. from mu, independently of
everything else, and order the limits s_(1) >= ... >= s_(N). Then for every k

    P(theta_eps <= s_(k)) >= 1 - P(Bin(N, eps) <= k - 1),

and on that event V(c) <= eps for every c > s_(k). No prediction model,
Lipschitz constant, or dimension enters. This bounds the feasible fraction,
not the operating limit: feasible designs may exist above U_eps on a set of
mu-measure at most eps, so L > U_eps is possible. The empirical-Bernstein
rejection test and the ladder of increasing stresses compute the individual
limits under noise (Appendix E.4).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binom


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1); got {value}")


def required_sample_size(eps: float, delta: float, *, k: int = 1) -> int:
    """Smallest N for which the k-th largest of N i.i.d. scores bounds theta_eps.

    Args:
        eps: Volume fraction in (0, 1).
        delta: Failure probability in (0, 1).
        k: Order index; k = 1 is the maximum. A larger k tolerates up to k - 1
            failed runs (scored +inf) at the cost of more samples.

    Returns:
        Smallest N >= k with P(Bin(N, eps) <= k - 1) <= delta.
    """
    _check_unit_interval("eps", eps)
    _check_unit_interval("delta", delta)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    lo = hi = k  # P(Bin(N, eps) <= k - 1) is nonincreasing in N
    while binom.cdf(k - 1, hi, eps) > delta:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binom.cdf(k - 1, mid, eps) > delta:
            lo = mid
        else:
            hi = mid
    return int(hi)


def order_statistic_upper(
    scores: np.ndarray, eps: float, delta: float
) -> tuple[float, int]:
    """U_eps = s_(k) for the largest k with P(Bin(N, eps) <= k - 1) <= delta.

    Args:
        scores: Scores of N designs drawn i.i.d. from mu, shape (N,). Use +inf for
            a failed run; it is the conservative label (it can only raise the
            bound). Use -inf only when infeasibility at every c is certain; under a
            noisy oracle a never-feasible verdict must carry its standard error or
            be scored +inf, never -inf with zero error.
        eps: Volume fraction in (0, 1).
        delta: Failure probability in (0, 1) for this bound alone.

    Returns:
        Tuple (U_eps, k). Returns (+inf, 0) when no k qualifies, which happens
        exactly when N < required_sample_size(eps, delta, k=1).
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"scores must be a non-empty 1-D array; got shape {s.shape}")
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN; use -inf or +inf")
    _check_unit_interval("eps", eps)
    _check_unit_interval("delta", delta)
    n = s.size
    # P(Bin(n, eps) <= j) is increasing in j, so the qualifying k = j + 1 form a
    # prefix and the count of qualifying j is the largest k.
    k = int(np.count_nonzero(binom.cdf(np.arange(n), n, eps) <= delta))
    if k == 0:
        return (np.inf, 0)
    return float(np.sort(s)[n - k]), k


def empirical_bernstein_deviation(
    sample_var: float, n: int, b: float, p: float
) -> float:
    """Upper deviation of a sample mean of n i.i.d. values in a range of width b.

    With probability at least 1 - p, ``mean - E[X] < t`` (Maurer and Pontil
    2009, Theorem 4, arXiv:0907.3740, applied to -X), with
    ``t = sqrt(2 V ln(2/p) / n) + 7 b ln(2/p) / (3 (n - 1))`` and V the
    unbiased sample variance. A test that declares ``E[X] > 0`` only when
    ``mean - t > 0`` therefore errs with probability at most p. Distribution
    free: only the range width b and independence are used (S3 construction
    S3-S1, where X is a clipped per-snapshot loading margin).

    Args:
        sample_var: Unbiased sample variance of the n values (ddof = 1).
        n: Sample size, at least 2.
        b: Width of the range the values are known to lie in; clipping from
            above shrinks it and only lowers the mean, so a positive clipped
            mean still implies a positive true mean.
        p: Error probability of the one-sided deviation, in (0, 1).

    Returns:
        The deviation t.

    Raises:
        ValueError: On n < 2, b <= 0, a negative variance, or p outside (0, 1).
    """
    _check_unit_interval("p", p)
    if n < 2:
        raise ValueError(f"n must be at least 2; got {n}")
    if b <= 0.0:
        raise ValueError(f"range width b must be positive; got {b}")
    if sample_var < 0.0:
        raise ValueError(f"sample variance must be non-negative; got {sample_var}")
    log_term = np.log(2.0 / p)
    return float(
        np.sqrt(2.0 * sample_var * log_term / n) + 7.0 * b * log_term / (3.0 * (n - 1))
    )


def ladder_upper_score(
    start: float,
    step: float,
    n_rungs: int,
    fires: Callable[[float], bool],
    c_max: float = 1.0,
) -> tuple[float, int]:
    """Smallest ladder coefficient at which a conservative infeasibility test fires.

    The rungs are ``start + step * j`` for j = 1, ..., n_rungs, clipped to
    c_max; the scan stops at the first clipped rung because later rungs
    repeat it. When ``fires(c)`` is a test that wrongly declares infeasibility
    with probability at most p per call, and infeasibility at c implies
    infeasibility above c, the returned coefficient is
    an upper score for the design's last feasible coefficient except with
    probability at most n_rungs * p.

    Args:
        start: Placement of the ladder (a rough estimate of the last feasible
            coefficient; it never enters the score).
        step: Positive rung spacing.
        n_rungs: Number of rungs, at least 1.
        fires: Conservative test, True when the coefficient is declared
            infeasible.
        c_max: Largest coefficient of the range.

    Returns:
        ``(score, n_tests)``: the first firing rung and the number of tests
        made; ``(inf, n_tests)`` when no rung fires.

    Raises:
        ValueError: On a non-positive step or fewer than one rung.
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive; got {step}")
    if n_rungs < 1:
        raise ValueError(f"n_rungs must be at least 1; got {n_rungs}")
    for j in range(1, n_rungs + 1):
        c = min(start + step * j, c_max)
        if fires(c):
            return float(c), j
        if c >= c_max:
            return float("inf"), j
    return float("inf"), n_rungs
