"""GP-UCB and differential evolution baselines (Appendix C.4).

Both maximize a design's observed last feasible stress over the unit design
box. One design evaluation is ``death_score``: a 9-point scan of the pass/fail
check on the unit stress range, then 8 bisection steps on the upper edge of
the widest feasible run (the window convention of ``brink.window_margins``
without the onset bisection). One evaluation costs at most 17 checks, each
charged the view's ``raw_evals_per_call``.

  * uncertified-gp-ucb: GP-UCB (Srinivas, Krause, Kakade and Seeger, ICML
    2010, arXiv:0912.3995) with a Matern-5/2 ARD kernel whose hyperparameters
    maximize the log marginal likelihood, beta_t from their Theorem 1 with
    |D| the candidate count, scaled down by 5 as in their experiments, and
    maximized over fresh uniform candidates each step.
  * uncertified-diff-evo: ``scipy.optimize.differential_evolution`` (Storn and
    Price 1997) at the library defaults, except no gradient polish and tol = 0,
    so a run stops on its budget; it restarts on a fresh stream when the
    population converges early.

Reporting matches bisection: the lower value is the best observed score, the
upper value sits one bisection step above it, and nothing is certified. A
design starts only when a full evaluation fits in the remaining budget.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import differential_evolution, minimize

from brink.arms import (
    ExistenceConfig,
    ExistenceReport,
    ExistenceResult,
)

ARM_GP_UCB = "uncertified-gp-ucb"
ARM_DIFF_EVO = "uncertified-diff-evo"
ADAPTIVE_ARMS = (ARM_GP_UCB, ARM_DIFF_EVO)

N_SCAN = 9  # window_margins.N_SCAN
BISECT_STEPS = 8  # window_margins.BISECT_STEPS
MAX_RAW_PER_DESIGN = N_SCAN + BISECT_STEPS
RESOLUTION = (1.0 / (N_SCAN - 1)) * 0.5**BISECT_STEPS
NO_WINDOW_SCORE = -1.0 / 200  # clamp: one report-grid step below c_min

BETA_SCALE = 0.2  # Srinivas et al. (2010) Section 6: beta_t scaled down by 5
HYPER_FIT_MAX_POINTS = 512  # random subset for the marginal-likelihood fit
HYPER_REFIT_GROWTH = 1.2  # refit when the data grow by 20 percent
LOG_LS_BOUNDS = (math.log(0.02), math.log(5.0))
LOG_SF2_BOUNDS = (math.log(0.05), math.log(20.0))
LOG_SN2_BOUNDS = (math.log(1e-6), math.log(1.0))
DE_SETTINGS = dict(
    strategy="best1bin", popsize=15, mutation=(0.5, 1.0), recombination=0.7,
    init="latinhypercube", polish=False, tol=0.0, atol=0.0, maxiter=10**6,
    updating="immediate",
)  # fmt: skip


def death_score(feasible: Callable[[float], bool]) -> float:
    """Observed last feasible stress of one design (unit c).

    Args:
        feasible: Raw predicate of the design on unit c.

    Returns:
        1.0 when the widest feasible scan run reaches c = 1; else the feasible
        end of the death bracket after 8 bisection steps (a point the
        predicate returned feasible); ``NO_WINDOW_SCORE`` when no scan point
        is feasible.
    """
    scan_c = np.linspace(0.0, 1.0, N_SCAN)
    scan_f = [bool(feasible(float(c))) for c in scan_c]
    if not any(scan_f):
        return NO_WINDOW_SCORE
    runs, i = [], 0
    while i < N_SCAN:
        if scan_f[i]:
            j = i
            while j + 1 < N_SCAN and scan_f[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    _, i1 = max(runs, key=lambda r: r[1] - r[0])  # first run wins ties
    if i1 == N_SCAN - 1:
        return 1.0
    lo, hi = float(scan_c[i1]), float(scan_c[i1 + 1])
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


class _BudgetExhaustedError(Exception):
    """No full design evaluation fits in the remaining raw budget."""


class _Objective:
    """Budgeted design scores with the sup-bisection report schedule."""

    def __init__(self, view, cfg: ExistenceConfig, arm: str, rng_noise) -> None:
        self.view, self.cfg, self.rng_noise = view, cfg, rng_noise
        self.cost = int(getattr(view, "raw_evals_per_call", 1))
        self.result = ExistenceResult(arm=arm)
        self.best, self.best_x, self.n_designs = NO_WINDOW_SCORE, None, 0
        self.next_report = cfg.report_every
        self.t0 = time.perf_counter()

    def can_start(self) -> bool:
        """A full evaluation (17 calls) fits in the remaining budget."""
        return self.result.raw_evals + MAX_RAW_PER_DESIGN * self.cost <= self.cfg.budget

    def _feasible(self, x: np.ndarray, c: float) -> bool:
        self.result.raw_evals += self.cost
        y = self.view.oracle(x, c, self.rng_noise, self.cfg.sigma)
        return bool(np.all(np.asarray(y) <= 0.0))

    def __call__(self, x) -> float:
        """Score of design x (clipped to the unit box); raises past the budget."""
        if not self.can_start():
            raise _BudgetExhaustedError
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        s = death_score(lambda c: self._feasible(x, c))
        self.n_designs += 1
        if s > self.best:
            self.best, self.best_x = s, [float(v) for v in x]
        self._report()
        return s

    @property
    def found(self) -> bool:
        return self.best >= 0.0

    def _bounds(self) -> tuple[float, float]:
        if not self.found:
            return 0.0, 1.0
        return self.best, min(self.best + RESOLUTION, 1.0)

    def _report(self, final: bool = False) -> None:
        """Scheduled reports due so far; ``final`` emits the rest up to N.

        The last design starts only when 17 calls fit, so a run can end up to
        16 calls short of N; its remaining reports carry the final state.
        """
        while (
            final or self.result.raw_evals >= self.next_report
        ) and self.next_report <= self.cfg.budget:
            lower, upper = self._bounds()
            self.result.reports.append(
                ExistenceReport(
                    report_index=len(self.result.reports) + 1,
                    calls=self.next_report // self.cost,
                    lower=lower,
                    upper=upper,
                    width=upper - lower,
                    has_witness=self.found,
                    has_certificate=False,
                    elapsed_s=time.perf_counter() - self.t0,
                    raw_evals=self.next_report,
                )  # fmt: skip
            )
            self.next_report += self.cfg.report_every

    def finish(self) -> ExistenceResult:
        """Final endpoints, the remaining reports, and the reported design."""
        self._report(final=True)
        self.result.final_lower, self.result.final_upper = self._bounds()
        self.result.final_witness_x = self.best_x if self.found else None
        return self.result


def _matern52(X1: np.ndarray, X2: np.ndarray, ls: np.ndarray, sf2: float):
    A, B = X1 / ls, X2 / ls
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    r = np.sqrt(5.0 * np.maximum(d2, 0.0))
    return sf2 * (1.0 + r + r * r / 3.0) * np.exp(-r)


def _neg_log_marginal(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    d = X.shape[1]
    ls, sf2, sn2 = np.exp(theta[:d]), math.exp(theta[d]), math.exp(theta[d + 1])
    K = _matern52(X, X, ls, sf2)
    K[np.diag_indices_from(K)] += sn2 + 1e-10
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e25
    alpha = cho_solve((L, True), y)
    return float(0.5 * y @ alpha + np.log(np.diag(L)).sum())


class _GP:
    """Exact GP regression with marginal-likelihood hyperparameters."""

    def __init__(self, x_dim: int, rng_fit: np.random.Generator) -> None:
        self.d, self.rng_fit = x_dim, rng_fit
        self.theta = np.r_[np.full(x_dim, math.log(0.3)), 0.0, math.log(1e-2)]
        self.n_fit, self.n_refits = 0, 0

    def _fit_hyper(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        idx = (
            self.rng_fit.choice(n, HYPER_FIT_MAX_POINTS, replace=False)
            if n > HYPER_FIT_MAX_POINTS
            else np.arange(n)
        )
        bounds = [LOG_LS_BOUNDS] * self.d + [LOG_SF2_BOUNDS, LOG_SN2_BOUNDS]
        default = np.r_[np.full(self.d, math.log(0.3)), 0.0, math.log(1e-2)]
        best = None
        for start in (self.theta, default):
            r = minimize(
                _neg_log_marginal, start, args=(X[idx], y[idx]),
                method="L-BFGS-B", bounds=bounds,
            )  # fmt: skip
            if best is None or r.fun < best.fun:
                best = r
        self.theta, self.n_fit = np.asarray(best.x), n
        self.n_refits += 1

    def update(self, X: np.ndarray, y_raw: np.ndarray) -> None:
        """Condition on all data; refit hyperparameters on the growth schedule."""
        self.y_mean = float(y_raw.mean())
        self.y_std = max(float(y_raw.std()), 1e-6)
        y = (y_raw - self.y_mean) / self.y_std
        if self.n_fit == 0 or X.shape[0] >= HYPER_REFIT_GROWTH * self.n_fit:
            self._fit_hyper(X, y)
        d = self.d
        self.ls, self.sf2 = np.exp(self.theta[:d]), math.exp(self.theta[d])
        K = _matern52(X, X, self.ls, self.sf2)
        K[np.diag_indices_from(K)] += math.exp(self.theta[d + 1]) + 1e-10
        self.X, self.L = X, cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), y)

    def predict(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and sd of the latent score (standardized scale)."""
        Ks = _matern52(Xq, self.X, self.ls, self.sf2)
        mu = Ks @ self.alpha
        v = solve_triangular(self.L, Ks.T, lower=True)
        var = np.maximum(self.sf2 - (v * v).sum(0), 1e-12)
        return mu, np.sqrt(var)


def _beta(n_candidates: int, t: int, delta: float) -> float:
    """Srinivas et al. (2010) Theorem 1 beta_t, scaled by ``BETA_SCALE``."""
    return BETA_SCALE * 2.0 * math.log(n_candidates * t * t * math.pi**2 / (6 * delta))


def _run_gp_ucb(view, cfg: ExistenceConfig, seed: int) -> tuple:
    f = _Objective(view, cfg, ARM_GP_UCB, np.random.default_rng([seed, 43]))
    rng_acq = np.random.default_rng([seed, 41])
    gp = _GP(int(view.spec.x_dim), np.random.default_rng([seed, 45]))
    X: list[np.ndarray] = []
    y: list[float] = []
    t = 0
    try:
        for x in rng_acq.uniform(size=(cfg.n_init, view.spec.x_dim)):
            y.append(f(x))
            X.append(x)
        while f.can_start():
            gp.update(np.asarray(X), np.asarray(y))
            t += 1
            cand = rng_acq.uniform(size=(cfg.n_candidates, view.spec.x_dim))
            mu, sd = gp.predict(cand)
            ucb = mu + math.sqrt(_beta(len(cand), t, cfg.delta)) * sd
            x = cand[int(np.argmax(ucb))]
            y.append(f(x))
            X.append(x)
    except _BudgetExhaustedError:
        pass
    info = {"n_designs": f.n_designs, "n_ucb_steps": t, "n_refits": gp.n_refits,
            "log_hyper": [float(v) for v in gp.theta]}  # fmt: skip
    return f.finish(), info


def _run_diff_evo(view, cfg: ExistenceConfig, seed: int) -> tuple:
    f = _Objective(view, cfg, ARM_DIFF_EVO, np.random.default_rng([seed, 53]))
    d = int(view.spec.x_dim)
    pop = int(DE_SETTINGS["popsize"]) * d
    first_pop_best: list[float] = []

    def neg_score(x) -> float:
        s = f(x)
        if f.n_designs == pop:
            first_pop_best.append(f.best)
        return -s

    restarts = 0
    while f.can_start():
        try:
            differential_evolution(
                neg_score, [(0.0, 1.0)] * d,
                rng=np.random.default_rng([seed, 47, restarts]), **DE_SETTINGS,
            )  # fmt: skip
        except _BudgetExhaustedError:
            break
        restarts += 1
    info = {"n_designs": f.n_designs, "n_restarts": restarts, "population": pop,
            "best_after_first_population":
                first_pop_best[0] if first_pop_best else None}  # fmt: skip
    return f.finish(), info


def run_adaptive_baseline(
    view, arm: str, cfg: ExistenceConfig, seed: int
) -> tuple[ExistenceResult, dict]:
    """Run one adaptive uncertified baseline on a raw-predicate view.

    Args:
        view: Predicate view with ``spec.x_dim``, ``oracle(x, c, rng, sigma)``
            (feasible iff every output is <= 0) and ``raw_evals_per_call``.
        arm: ``ARM_GP_UCB`` or ``ARM_DIFF_EVO``.
        cfg: Run constants; reads ``budget``, ``report_every``, ``n_init``,
            ``n_candidates``, ``delta``, and ``sigma``.
        seed: Run seed (arm streams [seed, 41/43/45] and [seed, 47/53]).

    Returns:
        The arm result (reports, endpoints, reported design) and run
        diagnostics (design count, refits or restarts).

    Raises:
        ValueError: For an unknown arm.
    """
    if arm == ARM_GP_UCB:
        return _run_gp_ucb(view, cfg, seed)
    if arm == ARM_DIFF_EVO:
        return _run_diff_evo(view, cfg, seed)
    raise ValueError(f"unknown adaptive baseline arm: {arm!r}")
