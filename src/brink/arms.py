"""Search rules for the operating limit theta = sup{c : F(c) nonempty}.

Model-based rules (all report through the same kernel bounds and tests):

  * existence-witness-focused: BRINK's proposed search (Appendix E.3). Each
    call goes to the stress one report step above the running lower bound,
    c_t = L + h_c, at the candidate design (past designs plus fresh uniform
    draws) with the smallest predicted worst constraint max_j mu_j(x, c_t).
  * existence-bracket-greedy: BRINK's variance-seeking rule:
    the largest posterior standard deviation over candidate pairs (x, c)
    with c in [L - h_c, U] whose design is not yet ruled out.
  * joint-config-existence: the CONFIG adaptation: among candidates whose
    lower bounds are all nonpositive, the one at the highest stress (CONFIG
    with the known objective -c).
  * random-joint-existence: uniform sampling with the same bounds.

Baselines without a guarantee:

  * uncertified-sup-bisection: bisection of stress on the pass/fail check for
    uniformly drawn designs, reporting the highest passing stress.
  * uncertified-direct-score: uniformly drawn designs, each scored by its own
    limit (``ScoreView``).

Raw-evaluation accounting: the budget counts raw evaluations (pass/fail
checks). An environment whose oracle call costs more than one raw evaluation
(a window oracle) sets ``last_raw_evals`` after each call; otherwise a call
costs one. The last call may overshoot the budget by less than one call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from brink.bracket import (
    SCAN_TOP_DOWN,
    ExistenceBracket,
    JointBands,
    compute_bracket,
    fit_joint_bands,
    midpoint_cover,
)
from brink.gp import beta_sqrt

ARM_BRACKET_GREEDY = "existence-bracket-greedy"
ARM_JOINT_CONFIG = "joint-config-existence"
ARM_RANDOM = "random-joint-existence"
ARM_SUP_BISECTION = "uncertified-sup-bisection"
ARM_WITNESS_FOCUSED = "existence-witness-focused"
ARM_DIRECT_SCORE = "uncertified-direct-score"
CERTIFIED_EXISTENCE_ARMS = (ARM_BRACKET_GREEDY, ARM_JOINT_CONFIG, ARM_RANDOM)
ALL_EXISTENCE_ARMS = (*CERTIFIED_EXISTENCE_ARMS, ARM_SUP_BISECTION)


@dataclass(frozen=True)
class ExistenceConfig:
    """Frozen run constants for the existence track.

    Attributes:
        mode: Surrogate mode (MODE_AFFINE for the reducible environment,
            MODE_RBF for the genuinely joint environment).
        budget: Raw-evaluation budget N (see the module docstring).
        n_init: Initial uniform design size.
        report_every: Calls between bracket reports.
        n_candidates: Acquisition candidate count per step.
        n_witness_candidates: Fresh witness-scan candidates per report.
        report_grid_size: Coefficient grid size K.
        cover_n_per_axis: Cover grid per axis for the emptiness certificate.
        cover_refine_factor: Local refinement factor for failing cover
            cells (sound two-level certificate; bracket.py).
        delta: Total envelope confidence budget (split delta / m).
        lengthscale: Joint-kernel lengthscale.
        lam: Regularization lambda.
        R: Known noise scale (0 deterministic).
        sigma: Oracle noise standard deviation.
        variant: "deterministic" or "noisy" (confidence-radius rule).
        bisect_depth: Per-x bisection depth of the uncertified arm.
        certificate_scan: SCAN_TOP_DOWN, SCAN_SMALLEST
            (smallest certified slice), or SCAN_OFF (no certificate: the
            upper endpoint stays c_max, where the
            certificate is data-limited). The two scans need registered
            nesting; neither covers the gaps between grid slices without it.
        cover_max_depth: 0 (default) keeps the two-level cover test; > 0
            uses the adaptive branch-and-bound test to this depth (needs
            SCAN_SMALLEST; ``bracket._slice_certifies_adaptive``).
        cover_max_cells: Per-slice cell-evaluation budget of the adaptive
            test.
        cover_second_order: False (default) keeps the first-order cell test;
            True ORs in the second-order clearance (needs cover_max_depth > 0;
            ``CONTEXT/research/second-order-clearance-lemma.md``, proof review
            pending).
    """

    mode: str
    budget: int = 300
    n_init: int = 20
    report_every: int = 10
    n_candidates: int = 2048
    n_witness_candidates: int = 2048
    report_grid_size: int = 201
    cover_n_per_axis: int = 61
    cover_refine_factor: int = 8
    delta: float = 0.1
    lengthscale: float = 0.3
    lam: float = 1e-4
    R: float = 0.0
    sigma: float = 0.0
    variant: str = "deterministic"
    bisect_depth: int = 8
    certificate_scan: str = SCAN_TOP_DOWN
    cover_max_depth: int = 0
    cover_max_cells: int = 8_000_000
    cover_second_order: bool = False


@dataclass
class ExistenceReport:
    """One scheduled report row."""

    report_index: int
    calls: int
    lower: float
    upper: float
    width: float
    has_witness: bool
    has_certificate: bool
    elapsed_s: float
    raw_evals: int = 0


@dataclass
class ExistenceResult:
    """Full arm output.

    ``n_resets`` counts off-event bracket crossings (each reset to the full
    range); ``n_config_steps`` counts acquisitions made by the CONFIG rule.
    """

    arm: str
    reports: list[ExistenceReport] = field(default_factory=list)
    raw_evals: int = 0
    n_resets: int = 0
    n_config_steps: int = 0
    final_lower: float = float("nan")
    final_upper: float = float("nan")
    final_witness_x: list[float] | None = None


def _bounds(env) -> tuple[float, float, int, int]:
    s = env.spec
    return float(s.c_min), float(s.c_max), int(s.x_dim), int(s.m)


def run_existence_arm(
    env,
    arm: str,
    cfg: ExistenceConfig,
    B: np.ndarray,
    seed: int,
    path_out: dict | None = None,
) -> ExistenceResult:
    """Run one arm to budget with scheduled bracket reports.

    Args:
        env: Environment with .oracle(x, c, rng, sigma) and .spec bounds.
        arm: One of ALL_EXISTENCE_ARMS, ARM_WITNESS_FOCUSED, or ARM_DIRECT_SCORE.
        cfg: Frozen constants.
        B: Per-constraint RKHS norm bounds (exact or calibration-fitted;
            the fitted case keeps the fitted-bound label in reports).
        seed: Run seed (dedicated streams per role).
        path_out: If given, receives the queried designs ``X``, stresses
            ``c``, and observations ``Y`` in call order (certified replay,
            certified replay). Does not change the run.

    Returns:
        ExistenceResult with per-report rows and the final running bracket.
    """
    if arm not in (*ALL_EXISTENCE_ARMS, ARM_WITNESS_FOCUSED, ARM_DIRECT_SCORE):
        raise ValueError(f"unknown existence arm: {arm!r}")
    if arm == ARM_SUP_BISECTION:
        return _run_sup_bisection(env, cfg, seed)
    if arm == ARM_DIRECT_SCORE:
        return _run_direct_score(env, cfg, seed)
    c_min, c_max, x_dim, m = _bounds(env)
    rng_acq = np.random.default_rng([seed, 11])
    rng_noise = np.random.default_rng([seed, 13])
    rng_wit = np.random.default_rng([seed, 17])
    c_grid = np.linspace(c_min, c_max, cfg.report_grid_size)
    h_c = float(c_grid[1] - c_grid[0])
    cover, rho = midpoint_cover(cfg.cover_n_per_axis, x_dim)
    delta_j = cfg.delta / m

    X_tr = np.zeros((cfg.budget, x_dim))
    c_tr = np.zeros(cfg.budget)
    Y = np.zeros((cfg.budget, m))
    run_lower, run_upper = c_min, c_max
    has_w_any, has_c_any = False, False
    result = ExistenceResult(arm=arm)
    bands = None
    t0 = time.perf_counter()

    init_x = rng_acq.uniform(size=(cfg.n_init, x_dim))
    init_c = rng_acq.uniform(c_min, c_max, size=cfg.n_init)

    call = 0
    while result.raw_evals < cfg.budget:
        call += 1
        if call <= cfg.n_init or bands is None:
            x = init_x[(call - 1) % cfg.n_init]
            cc = float(init_c[(call - 1) % cfg.n_init])
        elif arm == ARM_RANDOM:
            x = rng_acq.uniform(size=x_dim)
            cc = float(rng_acq.uniform(c_min, c_max))
        elif arm == ARM_WITNESS_FOCUSED:
            cc = min(run_lower + h_c, c_max)
            fresh = rng_acq.uniform(size=(cfg.n_candidates, x_dim))
            cand_x = np.concatenate([X_tr[: call - 1], fresh])
            x = cand_x[int(np.argmin(_witness_focus_score(bands, cand_x, cc)))]
        else:
            cand_x = rng_acq.uniform(size=(cfg.n_candidates, x_dim))
            lo = max(c_min, run_lower - h_c)
            cand_c = rng_acq.uniform(lo, run_upper, size=cfg.n_candidates)
            keep = bands.undetermined(cand_x, cand_c)
            if not keep.any():
                keep = np.ones(cfg.n_candidates, dtype=bool)
            if arm == ARM_BRACKET_GREEDY:
                score = np.where(keep, bands.sd_max(cand_x, cand_c), -np.inf)
            else:  # joint CONFIG: optimistic max-c
                score = np.where(keep, cand_c, -np.inf)
                result.n_config_steps += 1
            i = int(np.argmax(score))
            x, cc = cand_x[i], float(cand_c[i])
        y = env.oracle(x, cc, rng_noise, cfg.sigma)
        X_tr[call - 1], c_tr[call - 1], Y[call - 1] = x, cc, y
        result.raw_evals += int(getattr(env, "last_raw_evals", 1))

        betas_placeholder = [float(b) for b in B]
        bands = fit_joint_bands(
            X_tr[:call],
            c_tr[:call],
            Y[:call],
            mode=cfg.mode,
            lengthscale=cfg.lengthscale,
            lam=cfg.lam,
            B=B,
            R=cfg.R,
            betas=betas_placeholder,
            c_max=c_max,
        )
        bands.betas = [
            beta_sqrt(bands.models[j], delta_j, cfg.variant) for j in range(m)
        ]

        if call % cfg.report_every == 0 or result.raw_evals >= cfg.budget:
            wit = np.concatenate(
                [X_tr[:call], rng_wit.uniform(size=(cfg.n_witness_candidates, x_dim))]
            )
            br: ExistenceBracket = compute_bracket(
                bands,
                wit,
                c_grid,
                cover,
                rho,
                refine_factor=cfg.cover_refine_factor,
                scan=cfg.certificate_scan,
                max_depth=cfg.cover_max_depth,
                max_cells=cfg.cover_max_cells,
                second_order=cfg.cover_second_order,
            )
            if br.has_witness:
                if br.lower >= run_lower and br.witness_x is not None:
                    result.final_witness_x = [float(v) for v in br.witness_x]
                run_lower = max(run_lower, br.lower)
                has_w_any = True
            if br.has_certificate:
                run_upper = min(run_upper, br.upper)
                has_c_any = True
            if run_upper < run_lower:  # off-event symptom; keep sound order
                run_lower, run_upper = c_min, c_max
                has_w_any = has_c_any = False
                result.n_resets += 1
            result.reports.append(
                ExistenceReport(
                    report_index=len(result.reports) + 1,
                    calls=call,
                    lower=run_lower,
                    upper=run_upper,
                    width=run_upper - run_lower,
                    has_witness=has_w_any,
                    has_certificate=has_c_any,
                    elapsed_s=time.perf_counter() - t0,
                    raw_evals=result.raw_evals,
                )
            )
    result.final_lower, result.final_upper = run_lower, run_upper
    if path_out is not None:
        path_out.update(X=X_tr[:call].copy(), c=c_tr[:call].copy(), Y=Y[:call].copy())
    return result


def _witness_focus_score(bands: JointBands, X: np.ndarray, c: float) -> np.ndarray:
    """Witness-focused acquisition score at slice c: max_j mu_j (lower is better)."""
    return np.max([bands._mu_sd(j, X, c)[0] for j in range(len(bands.models))], axis=0)


def _run_sup_bisection(env, cfg: ExistenceConfig, seed: int) -> ExistenceResult:
    """Uncertified baseline: max over sampled x of per-x bisected crossing.

    Each predicate decision costs ``env.raw_evals_per_call`` raw evaluations
    (1 unless the view says otherwise); the budget, the reports, and the
    stop rule are all in raw evaluations, so a replication-hardened predicate
    (S3 arms 5-6) never exceeds the matched raw budget.
    """
    c_min, c_max, x_dim, _ = _bounds(env)
    rng = np.random.default_rng([seed, 23])
    rng_noise = np.random.default_rng([seed, 29])
    result = ExistenceResult(arm=ARM_SUP_BISECTION)
    cost = int(getattr(env, "raw_evals_per_call", 1))
    best = c_min
    found_any = False
    calls = 0
    t0 = time.perf_counter()
    step = (c_max - c_min) * 0.5**cfg.bisect_depth
    next_report = cfg.report_every

    def can_observe() -> bool:
        return result.raw_evals + cost <= cfg.budget

    def observe_feasible(x: np.ndarray, c: float) -> bool:
        nonlocal calls
        calls += 1
        result.raw_evals += cost
        y = env.oracle(x, c, rng_noise, cfg.sigma)
        return bool(np.all(y <= 0.0))

    while can_observe():
        x = rng.uniform(size=x_dim)
        # Probe above the incumbent: only improvements matter for a sup.
        probe_lo = max(best, c_min)
        if not observe_feasible(x, probe_lo):
            pass  # x cannot improve the incumbent (by observed predicate)
        else:
            lo, hi = probe_lo, c_max
            if can_observe() and observe_feasible(x, hi):
                lo = hi
            else:
                for _ in range(cfg.bisect_depth):
                    if not can_observe():
                        break
                    mid = 0.5 * (lo + hi)
                    if observe_feasible(x, mid):
                        lo = mid
                    else:
                        hi = mid
            best = max(best, lo)
            found_any = True
        while result.raw_evals >= next_report and next_report <= cfg.budget:
            lower = best if found_any else c_min
            upper = min(best + step, c_max) if found_any else c_max
            result.reports.append(
                ExistenceReport(
                    report_index=len(result.reports) + 1,
                    calls=next_report // cost,
                    lower=lower,
                    upper=upper,
                    width=upper - lower,
                    has_witness=found_any,
                    has_certificate=False,
                    elapsed_s=time.perf_counter() - t0,
                    raw_evals=next_report,
                )
            )
            next_report += cfg.report_every
    result.final_lower = best if found_any else c_min
    result.final_upper = min(best + step, c_max) if found_any else c_max
    return result


def _run_direct_score(env, cfg: ExistenceConfig, seed: int) -> ExistenceResult:
    """Uncertified baseline: running max of the reduced score over samples.

    Mandatory under gate G1 whenever the coefficient is eliminable from the
    oracle. One call per design returns that design's score, so the estimator
    is ``max_x score(x)`` over uniformly sampled designs.

    Endpoint convention, matching the uncertified bisection baseline: the
    lower endpoint is the running maximum, which is sound because the design
    attaining it is genuinely feasible there; the upper endpoint is one
    report-grid step above it, which is the uncertified claim that the
    sampled maximum has located the supremum. Failure to sample a better
    design proves nothing, so the upper endpoint carries no guarantee.
    """
    c_min, c_max, x_dim, _ = _bounds(env)
    rng = np.random.default_rng([seed, 31])
    result = ExistenceResult(arm=ARM_DIRECT_SCORE)
    grid_step = (c_max - c_min) / (cfg.report_grid_size - 1)
    best = c_min
    found_any = False
    next_report = cfg.report_every
    t0 = time.perf_counter()

    while result.raw_evals < cfg.budget:
        x = rng.uniform(size=x_dim)
        before = getattr(env, "n_raw_evals", None)
        s = float(env.score(x))
        if before is None:
            result.raw_evals += 1
        else:
            result.raw_evals += env.n_raw_evals - before
        if s >= best:
            best = max(best, s)
            result.final_witness_x = [float(v) for v in np.asarray(x, dtype=float)]
            found_any = True
        while result.raw_evals >= next_report and next_report <= cfg.budget:
            lower = best if found_any else c_min
            upper = min(c_max, lower + grid_step) if found_any else c_max
            result.reports.append(
                ExistenceReport(
                    report_index=len(result.reports) + 1,
                    calls=next_report,
                    raw_evals=next_report,
                    lower=lower,
                    upper=upper,
                    width=upper - lower,
                    has_witness=found_any,
                    has_certificate=False,
                    elapsed_s=time.perf_counter() - t0,
                )
            )
            next_report += cfg.report_every

    result.final_lower = best if found_any else c_min
    result.final_upper = (
        min(c_max, result.final_lower + grid_step) if found_any else c_max
    )
    return result
