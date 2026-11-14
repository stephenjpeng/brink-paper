"""Views of a simulator behind the search-rule interface.

The search rules (``arms.py``) need ``spec`` (``c_min``, ``c_max``, ``x_dim``,
``m``) and ``oracle(x, c, rng, sigma)``, and count raw evaluations (pass/fail
checks) against the budget. A window oracle spends about 10 to 18 checks per
call (at most 32), so :class:`WindowExistenceView` reports each call's exact
cost from the environment's window counters. Bisection gets the pass/fail
check itself through :class:`RawPredicateView`, one raw evaluation per query,
and direct limit search gets each design's limit through :class:`ScoreView`.
All views work on the unit stress coordinate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExistenceSpec:
    """Bounds the existence arms read from ``env.spec``."""

    x_dim: int
    m: int
    c_min: float = 0.0
    c_max: float = 1.0


class WindowExistenceView:
    """A portfolio window environment with per-call raw-evaluation cost.

    Args:
        env: Portfolio environment with ``x_dim``, ``m``,
            ``oracle(x, c, rng, sigma)`` on the unit coefficient, and
            ``window_counters()`` reporting cumulative ``n_raw_evals``.
    """

    def __init__(self, env) -> None:
        """Wrap a portfolio window environment."""
        self.env = env
        self.spec = ExistenceSpec(x_dim=int(env.x_dim), m=int(env.m))
        self.last_raw_evals = 0

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """One window call; sets ``last_raw_evals`` to its raw cost.

        Args:
            x: Unit design point.
            c: Unit coefficient.
            rng: Random generator (environment noise, if any).
            sigma: Environment noise argument, passed through.

        Returns:
            Standardized window channels, shape (m,).
        """
        before = self.env.window_counters()["n_raw_evals"]
        y = self.env.oracle(x, c, rng, sigma)
        self.last_raw_evals = self.env.window_counters()["n_raw_evals"] - before
        return y


class RawPredicateView:
    """The raw feasibility predicate as a one-channel oracle (1 raw eval each).

    Args:
        raw_feasible: Predicate (x_unit, c_unit) -> bool, or
            (x_unit, c_unit, rng) -> bool when ``needs_rng``; a crash counts
            as infeasible (the window-margin convention).
        x_dim: Design-space dimension.
        needs_rng: Pass the call RNG to the predicate (a stochastic predicate
            such as the hosting K-snapshot mean draws its snapshots from it).
        raw_evals_per_call: Raw evaluations charged per predicate query (a
            K'-snapshot predicate on an oracle whose raw unit is
            K snapshots costs K' / K; the S3 replication-hardened baselines).
    """

    def __init__(
        self,
        raw_feasible: Callable[..., bool],
        x_dim: int,
        *,
        needs_rng: bool = False,
        raw_evals_per_call: int = 1,
    ):
        """Bind the raw predicate."""
        if int(raw_evals_per_call) < 1:
            raise ValueError("raw_evals_per_call must be a positive integer")
        self.raw_feasible = raw_feasible
        self.spec = ExistenceSpec(x_dim=int(x_dim), m=1)
        self.needs_rng = needs_rng
        self.raw_evals_per_call = int(raw_evals_per_call)
        self.n_raw_evals = 0

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Return [-1] when the raw predicate holds, else [1].

        Args:
            x: Unit design point.
            c: Unit coefficient.
            rng: Call RNG, passed to the predicate when ``needs_rng``;
                otherwise unused.
            sigma: Unused.

        Returns:
            Array of shape (1,).

        Raises:
            ValueError: When ``needs_rng`` and rng is None.
        """
        if not self.needs_rng:
            ok = self.raw_feasible(x, float(c))
        elif rng is None:
            raise ValueError("stochastic raw predicate requires a call RNG")
        else:
            ok = self.raw_feasible(x, float(c), rng)
        self.n_raw_evals += self.raw_evals_per_call
        return np.array([-1.0 if ok else 1.0])


class ScoreView:
    """The reduced score c_off(x) as a one-call-per-design oracle.

    Gate G1 makes a direct reduced-score baseline mandatory whenever the
    coefficient is eliminable from the oracle. On such an oracle one call per
    design determines that design's whole slice answer, so the direct
    estimator is ``max_x score(x)`` over sampled designs.

    Args:
        score_fn: Map from a unit design point to its unit-coefficient
            score, that is the largest coefficient at which it stays
            feasible.
        x_dim: Design-space dimension.
        raw_evals_per_call: Raw-evaluation cost of one score query.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], float],
        x_dim: int,
        *,
        raw_evals_per_call: int = 1,
    ):
        """Bind the score function."""
        self.score_fn = score_fn
        self.spec = ExistenceSpec(x_dim=int(x_dim), m=1)
        self.raw_evals_per_call = int(raw_evals_per_call)
        self.n_raw_evals = 0

    def score(self, x: np.ndarray) -> float:
        """Return the design's score and charge its raw-evaluation cost."""
        self.n_raw_evals += self.raw_evals_per_call
        return float(self.score_fn(x))
