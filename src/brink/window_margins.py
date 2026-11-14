"""Window observations of a pass/fail simulator (Appendix C.2, Section 5).

One oracle call at (x, c) evaluates the pass/fail check at a 9-point scan of
the unit stress range, brackets the feasible window, bisects each end 8 steps
(at most 32 checks per call), and returns the window channels

    g_lo = c_on(x) - c    and    g_hi = c - c_off(x),

both exactly affine in c, so feasible iff both are <= 0. A design with no
feasible window returns c_on = c_max + 0.25 and c_off = c_min - 0.25. If the
scan finds several windows, the widest is used (first on ties) and the event
is counted. Studies whose designs are feasible from the lowest stress use
the upper-window channel alone (m = 1); in that mode a design with no window
may instead report c_off one report-grid step below c_min.

Window channels are standardized by scale only and not clipped: the no-window
constants bound the raw channels in [-1.0, 1.25], and clipping would break
the affinity in c.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Construction constants.
N_SCAN = 9
BISECT_STEPS = 8
MAX_RAW_EVALS = 32
NO_WINDOW_C_ON = 1.25  # c_max_unit + 0.25
NO_WINDOW_C_OFF = -0.25  # c_min_unit - 0.25

FeasiblePredicate = Callable[[float], bool]
RawFeasible = Callable[[np.ndarray, float], bool]


@dataclass(frozen=True)
class WindowResult:
    """Feasibility window of one design column in unit c.

    Attributes:
        c_on: Window onset (unit c); NO_WINDOW_C_ON when no window found.
        c_off: Window death (unit c); NO_WINDOW_C_OFF when no window found.
        n_raw_evals: Raw predicate evaluations spent (at most 32).
        multi_window: True when the coarse scan saw a non-contiguous
            feasible pattern (the widest window was kept).
    """

    c_on: float
    c_off: float
    n_raw_evals: int
    multi_window: bool


def extract_window(
    feasible: FeasiblePredicate, *, no_window_c_off: float = NO_WINDOW_C_OFF
) -> WindowResult:
    """Scan-and-bisect window extraction.

    9-point coarse scan of the raw feasibility predicate across the unit
    c range (endpoints included), bracket the feasible window, bisect each
    endpoint 8 steps; hard cap of 32 raw evaluations per call.

    Args:
        feasible: Raw feasibility predicate on unit c (crash = infeasible,
            handled by the caller's predicate).
        no_window_c_off: Death endpoint returned when no window is found.
            The default is the m=2 constant -0.25. prefix mode passes
            minus one report-grid step instead, making the death field
            near-continuous at the feasibility boundary.

    Returns:
        WindowResult with the window endpoints and evaluation accounting.

    Raises:
        RuntimeError: If the 32-raw-evaluation cap would be
            exceeded (impossible by construction: 9 + 8 + 8 = 25).
    """
    count = 0

    def feas(c: float) -> bool:
        nonlocal count
        count += 1
        if count > MAX_RAW_EVALS:
            raise RuntimeError(
                f"raw-evaluation cap exceeded ({MAX_RAW_EVALS} per oracle call)"
            )
        return bool(feasible(float(c)))

    def bisect(lo: float, hi: float, lo_feasible: bool) -> float:
        """Bisect the feasibility edge between lo and hi (8 steps)."""
        for _ in range(BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if feas(mid) == lo_feasible:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    scan_c = np.linspace(0.0, 1.0, N_SCAN)
    scan_f = np.array([feas(c) for c in scan_c])
    if not scan_f.any():
        return WindowResult(NO_WINDOW_C_ON, float(no_window_c_off), count, False)

    # Contiguous feasible runs as (first_index, last_index) pairs.
    runs: list[tuple[int, int]] = []
    i = 0
    while i < N_SCAN:
        if scan_f[i]:
            j = i
            while j + 1 < N_SCAN and scan_f[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    multi = len(runs) > 1
    # Widest window (scan points are equally spaced, so run length is
    # width); first run wins ties.
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])

    c_on = 0.0 if i0 == 0 else bisect(float(scan_c[i0 - 1]), float(scan_c[i0]), False)
    c_off = (
        1.0
        if i1 == N_SCAN - 1
        else bisect(float(scan_c[i1]), float(scan_c[i1 + 1]), True)
    )
    return WindowResult(c_on, c_off, count, multi)


def window_channels(
    c_on: float, c_off: float, c: float, *, prefix_only: bool = False
) -> np.ndarray:
    """Raw window-margin channels at a unit coefficient.

    Args:
        c_on: Window onset in unit c.
        c_off: Window death in unit c.
        c: Query coefficient in unit c.
        prefix_only: When True (m = 1 prefix mode, c_on
            identically c_min), return only the g_hi channel.

    Returns:
        Array (g_lo, g_hi) of shape (2,), or (g_hi,) of shape (1,) in
        prefix mode; feasible iff every channel is <= 0.
    """
    g_hi = c - c_off
    if prefix_only:
        return np.array([g_hi])
    return np.array([c_on - c, g_hi])


class WindowMarginOracle:
    """One-call-per-window oracle wrapper with sidecar accounting.

    One oracle call = one window computation (no caching across calls, so
    all arms consume matched accounting). Counters accumulate per instance
    and are reported in the run sidecar (raw-evaluations-per-call
    multiplier, multi-window events).

    Attributes:
        raw_feasible: Raw predicate on (x_unit, c_unit).
        prefix_only: M = 1 prefix mode flag.
        n_calls: Oracle calls made.
        n_raw_evals: Raw predicate evaluations spent.
        n_multi_window: Calls whose coarse scan saw multiple windows.
    """

    def __init__(
        self,
        raw_feasible: RawFeasible,
        *,
        prefix_only: bool = False,
        prefix_no_window_report_grid_step: float | None = None,
    ):
        """Bind the raw feasibility predicate.

        Args:
            raw_feasible: Predicate (x_unit, c_unit) -> bool; a crashed raw
                oracle must be mapped to False by the caller.
            prefix_only: When True, channels are (g_hi,) only.
            prefix_no_window_report_grid_step: prefix-mode clamp. When
                given, a no-window column uses
                ``c_off = -prefix_no_window_report_grid_step`` instead of
                the m=2 constant -0.25. Invalid outside prefix mode.

        Raises:
            ValueError: When the clamp is nonpositive or requested
                outside prefix mode.
        """
        if prefix_no_window_report_grid_step is not None:
            if not prefix_only:
                raise ValueError("P5 no-window clamp requires prefix_only=True")
            if prefix_no_window_report_grid_step <= 0.0:
                raise ValueError("prefix no-window report-grid step must be positive")
        self.raw_feasible = raw_feasible
        self.prefix_only = prefix_only
        self.prefix_no_window_report_grid_step = prefix_no_window_report_grid_step
        self.n_calls = 0
        self.n_raw_evals = 0
        self.n_multi_window = 0

    @property
    def m(self) -> int:
        """Number of window channels (1 in prefix mode, else 2)."""
        return 1 if self.prefix_only else 2

    def window(self, x: np.ndarray) -> WindowResult:
        """Extract the feasibility window of one design column.

        Args:
            x: Unit design point.

        Returns:
            WindowResult; counters are updated.
        """
        no_window_c_off = (
            -self.prefix_no_window_report_grid_step
            if self.prefix_no_window_report_grid_step is not None
            else NO_WINDOW_C_OFF
        )
        result = extract_window(
            lambda c: self.raw_feasible(x, c), no_window_c_off=no_window_c_off
        )
        self.n_calls += 1
        self.n_raw_evals += result.n_raw_evals
        self.n_multi_window += int(result.multi_window)
        return result

    def raw_channels(self, x: np.ndarray, c: float) -> np.ndarray:
        """Raw (unstandardized) window channels at unit (x, c).

        Args:
            x: Unit design point.
            c: Unit coefficient.

        Returns:
            Raw channels of shape (m,), exactly affine in c.
        """
        w = self.window(x)
        return window_channels(w.c_on, w.c_off, c, prefix_only=self.prefix_only)

    def counters(self) -> dict:
        """Sidecar accounting: calls, raw evaluations, multiplier, events."""
        return {
            "n_calls": self.n_calls,
            "n_raw_evals": self.n_raw_evals,
            "raw_evals_per_call": (
                self.n_raw_evals / self.n_calls if self.n_calls else 0.0
            ),
            "n_multi_window": self.n_multi_window,
        }
