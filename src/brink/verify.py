"""Check layer (Appendix C.7): test a returned design and adjust its reported level.

The layer takes a design x and level L and spends at most ``n_max`` checks: a
check at L; on a pass, steps up by doubling increments (L + (2^k - 1) step0,
capped at c_max); on a fail, steps down the same way (floored at c_min); then
bisects the pass/fail bracket while checks remain and it is wider than
``resolution``. The tested level is the highest stress with a passing check.
The layer never sees the prediction model, so the same code serves every
method and study. It gives no global upper bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdjustResult:
    """Outcome of the check layer on one design.

    Attributes:
        tested_lower: Highest coefficient with a passing check (unit c), or
            None when no check passed (no tested design).
        n_checks: System checks spent, at most ``n_max``.
        trace: Every (c, passed) in check order.
    """

    tested_lower: float | None
    n_checks: int
    trace: list[tuple[float, bool]]


def verify_and_adjust(
    x: np.ndarray,
    lower: float,
    check: Callable[[np.ndarray, float], bool],
    n_max: int,
    c_min: float,
    c_max: float,
    step0: float,
    resolution: float,
) -> AdjustResult:
    """Check layer: test design x near its reported level and adjust the level.

    Args:
        x: Proposed design (unit coordinates).
        lower: Proposed endpoint L (unit c, in [c_min, c_max]).
        check: System check ``check(x, c) -> passed``; each call costs one
            check of the reserve.
        n_max: Check reserve (at least 1).
        c_min: Coefficient floor; a failing check here ends the run with no
            tested design.
        c_max: Coefficient ceiling; a passing check here ends the run.
        step0: First step Delta_0 of the doubling search.
        resolution: Bisection stops once the bracket is at most this wide.

    Returns:
        The tested endpoint, the checks spent, and the trace.
    """
    trace: list[tuple[float, bool]] = []

    def passed(c: float) -> bool:
        ok = bool(check(x, c))
        trace.append((float(c), ok))
        return ok

    L = float(lower)
    c_pass: float | None = None
    c_fail: float | None = None
    k = 1
    if passed(L):
        c_pass = L
        while len(trace) < n_max and c_pass < c_max:
            c = min(L + (2**k - 1) * step0, c_max)
            if not passed(c):
                c_fail = c
                break
            c_pass, k = c, k + 1
    else:
        c_fail = L
        while len(trace) < n_max and c_fail > c_min:
            c = max(L - (2**k - 1) * step0, c_min)
            if passed(c):
                c_pass = c
                break
            c_fail, k = c, k + 1
    if c_pass is not None and c_fail is not None:
        while len(trace) < n_max and c_fail - c_pass > resolution:
            mid = 0.5 * (c_pass + c_fail)
            if passed(mid):
                c_pass = mid
            else:
                c_fail = mid
    return AdjustResult(tested_lower=c_pass, n_checks=len(trace), trace=trace)
