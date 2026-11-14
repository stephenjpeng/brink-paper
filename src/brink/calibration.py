"""Calibration recipe that fixes each study's model constants (Appendix C.1).

Identical for every study; the constants are frozen before the reported runs
(``configs/frozen/``). Calibration points never enter a search run.

- Standardization is scale-only: each raw channel is divided by its
  calibration interquartile range / 1.349. The median is recorded but never
  subtracted, so zero remains the feasibility boundary.
- The lengthscale is the median pairwise distance of the 512 calibration
  designs times a multiplier from {0.5, 1.0, 2.0} chosen by 5-fold
  cross-validation.
- The norm bound B_j is twice the RKHS norm of the kernel-ridge fit to the
  calibration set; the per-slice bound B_slice_j is recorded for reference.
- In the noisy studies the noise scale R is 1.2 times the pooled standard
  deviation of 100 replicate pairs, and lambda = R^2.
- Crashed points take the crash margin (+3.0 per channel) after
  standardization, and raw-margin channels are clipped to +-3; window
  channels are not clipped (``clip=False``).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from brink.gp import fit_gp, posterior
from brink.kernels import rbf

logger = logging.getLogger(__name__)

# Constants; never tune these.
N_SOBOL = 256
N_UNIFORM = 256
SOBOL_SEED = 9001
UNIFORM_SEED = 9002
SPLIT_SEED = 9003
RIDGE_LAM = 1.0e-6
NORM_INFLATION = 2.0
LENGTHSCALE_MULTIPLIERS = (0.5, 1.0, 2.0)
N_CV_FOLDS = 5
N_SLICE_ANCHORS = 8
HOLDOUT_FRACTION = 0.2
R2_MIN = 0.5
IQR_TO_SD = 1.349
# Noise-scale recipe: 100 replicate pairs at 50
# random points per noise knob; R(knob) = 1.2 x sd((difference)/sqrt(2)).
REPLICATE_N_POINTS = 50
REPLICATE_N_PAIRS = 100
REPLICATE_POINT_SEED = 9004
R_INFLATION = 1.2

RawOracle = Callable[[np.ndarray, float], np.ndarray | None]
ReplicatedOracle = Callable[[np.ndarray, float], np.ndarray]


def measure_noise_scale(
    replicated_oracle: ReplicatedOracle,
    x_dim: int,
    scale: np.ndarray,
) -> dict:
    """Measure the frozen noise scale R of one noise knob (step 5).

    100 replicate pairs at 50 random points (2 pairs per point, drawn from
    the point seed 9004 over unit X x C): each pair is two
    independent oracle calls at the identical point; the frozen noise scale
    is R = 1.2 times the empirical sd of (replicate difference) / sqrt(2),
    pooled over pairs and standardized channels. The caller binds the
    oracle to its noise knob and its replicate RNG stream.

    Args:
        replicated_oracle: Callable (x_unit, c_unit) -> raw margins (m,);
            every call must draw fresh noise from the bound stream.
        x_dim: Design-space dimension.
        scale: Frozen per-channel standardization scale, shape (m,).

    Returns:
        Dict with the frozen R, lam = R^2, per-channel sd, and accounting.
    """
    rng = np.random.default_rng(REPLICATE_POINT_SEED)
    pts_x = rng.uniform(size=(REPLICATE_N_POINTS, x_dim))
    pts_c = rng.uniform(size=REPLICATE_N_POINTS)
    pairs_per_point = REPLICATE_N_PAIRS // REPLICATE_N_POINTS
    diffs = []
    for i in range(REPLICATE_N_POINTS):
        for _ in range(pairs_per_point):
            ya = np.asarray(replicated_oracle(pts_x[i], float(pts_c[i])))
            yb = np.asarray(replicated_oracle(pts_x[i], float(pts_c[i])))
            diffs.append((ya - yb) / scale)
    d = np.asarray(diffs) / np.sqrt(2.0)
    pooled_sd = float(d.std(ddof=1))
    R = R_INFLATION * pooled_sd
    return {
        "R": R,
        "lam": R * R,
        "pooled_sd": pooled_sd,
        "per_channel_sd": [float(v) for v in d.std(axis=0, ddof=1)],
        "n_points": REPLICATE_N_POINTS,
        "n_pairs": len(diffs),
        "point_seed": REPLICATE_POINT_SEED,
        "inflation": R_INFLATION,
    }


@dataclass(frozen=True)
class FrozenCalibration:
    """Frozen calibration constants for one environment.

    Attributes:
        experiment: Experiment name the constants belong to.
        x_dim: Design-space dimension.
        m: Number of margin channels.
        scaler_median: Per-channel calibration median (recorded, never
            subtracted; see the module interpretation notes).
        scaler_scale: Per-channel scale IQR / 1.349 dividing raw margins.
        crash_margin: Standardized margin returned on oracle crash.
        n_crashed: Crashed calibration evaluations out of 512.
        median_pairwise_distance_x: Median pairwise distance of the 512
            calibration x-points.
        lengthscale_multiplier: Selected multiplier from {0.5, 1.0, 2.0}.
        lengthscale: Frozen RBF lengthscale on X.
        cv_rmse_by_multiplier: 5-fold kernel-ridge RMSE per multiplier.
        B: Frozen inflated joint-RKHS norm bound per channel.
        B_slice: Frozen inflated per-slice norm bound per channel.
        R: Frozen noise scale (0 for deterministic environments).
        lam: Frozen GP regularization (1e-6 deterministic; R^2 stochastic).
        r2_holdout: Held-out 20% calibration R^2 per channel.
        r2_ok: True when every channel meets the R^2 >= 0.5 acceptance.
        seeds: The seeds used (sobol, uniform, split).
        clip: Whether standardized margins were clipped (False for window
            channels per amendment P2).
    """

    experiment: str
    x_dim: int
    m: int
    scaler_median: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    crash_margin: float
    n_crashed: int
    median_pairwise_distance_x: float
    lengthscale_multiplier: float
    lengthscale: float
    cv_rmse_by_multiplier: dict[str, float]
    B: tuple[float, ...]
    B_slice: tuple[float, ...]
    R: float
    lam: float
    r2_holdout: tuple[float, ...]
    r2_ok: bool
    seeds: dict[str, int]
    clip: bool = True


def calibration_points(x_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw the 512 calibration points on unit X x C.

    256 scrambled-Sobol points (seed 9001) plus 256 uniform points
    (seed 9002), in that order, over the (x_dim + 1)-dimensional unit box;
    the last coordinate is the unit coefficient.

    Args:
        x_dim: Design-space dimension.

    Returns:
        Tuple (X_unit, c_unit) of shapes (512, x_dim) and (512,). The first
        256 rows are the Sobol block.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sob = qmc.Sobol(d=x_dim + 1, scramble=True, seed=SOBOL_SEED)
        u_sobol = sob.random(N_SOBOL)
    rng = np.random.default_rng(UNIFORM_SEED)
    u_uniform = rng.uniform(size=(N_UNIFORM, x_dim + 1))
    u = np.vstack([u_sobol, u_uniform])
    return u[:, :x_dim], u[:, x_dim]


def evaluate_raw(
    raw_oracle: RawOracle, X: np.ndarray, c: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the raw oracle once per calibration point.

    Args:
        raw_oracle: Callable (x_unit, c_unit) -> raw margins (m,), or None
            when the oracle crashes (e.g. power-flow non-convergence).
        X: Unit design points, shape (n, x_dim).
        c: Unit coefficients, shape (n,).
        m: Number of margin channels.

    Returns:
        Tuple (Y_raw, crashed): raw margins with NaN rows at crashes,
        shape (n, m), and the boolean crash mask, shape (n,).
    """
    n = X.shape[0]
    Y = np.full((n, m), np.nan)
    crashed = np.zeros(n, dtype=bool)
    for i in range(n):
        out = raw_oracle(X[i], float(c[i]))
        if out is None:
            crashed[i] = True
        else:
            Y[i] = np.asarray(out, dtype=float)
    return Y, crashed


def fit_scalers(
    Y_raw: np.ndarray, crashed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and scale (IQR / 1.349) from converged points.

    Args:
        Y_raw: Raw margins, shape (n, m), NaN rows at crashes.
        crashed: Boolean crash mask, shape (n,).

    Returns:
        Tuple (median, scale), each of shape (m,).

    Raises:
        ValueError: When a channel's IQR is zero (degenerate channel).
    """
    good = Y_raw[~crashed]
    median = np.median(good, axis=0)
    iqr = np.quantile(good, 0.75, axis=0) - np.quantile(good, 0.25, axis=0)
    if np.any(iqr <= 0.0):
        raise ValueError(f"degenerate margin channel: IQR per channel = {iqr}")
    return median, iqr / IQR_TO_SD


def standardize(
    Y_raw: np.ndarray,
    crashed: np.ndarray,
    scale: np.ndarray,
    crash_margin: float,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Scale-only standardization with crash convention and clipping.

    Standardized values are clipped to [-crash_margin, +crash_margin];
    the clip never crosses zero, so
    the feasible set is unchanged. Window-margin channels pass
    ``clip=False``.

    Args:
        Y_raw: Raw margins, shape (n, m), NaN rows at crashes.
        crashed: Boolean crash mask, shape (n,).
        scale: Per-channel scale IQR / 1.349, shape (m,).
        crash_margin: Standardized margin assigned to crashed points and
            the symmetric clipping bound.
        clip: Whether to clip standardized values (raw-margin convention).

    Returns:
        Standardized margins, shape (n, m), in [-crash_margin, crash_margin]
        when ``clip`` is True.
    """
    Y = Y_raw / scale[None, :]
    if clip:
        Y = np.clip(Y, -crash_margin, crash_margin)
    Y[crashed] = crash_margin
    return Y


def median_pairwise_distance(X: np.ndarray) -> float:
    """Median pairwise Euclidean distance between calibration x-points.

    Args:
        X: Unit design points, shape (n, x_dim).

    Returns:
        Median over all n * (n - 1) / 2 pairwise distances.
    """
    sq = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2.0 * (X @ X.T)
    np.maximum(sq, 0.0, out=sq)
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.median(np.sqrt(sq[iu])))


def _cv_rmse(
    X: np.ndarray,
    c: np.ndarray,
    Y: np.ndarray,
    lengthscale: float,
    folds: list[np.ndarray],
) -> float:
    """5-fold kernel-ridge RMSE (lam 1e-6) pooled over folds and channels."""
    sq_errors: list[np.ndarray] = []
    for fold in folds:
        test = np.zeros(X.shape[0], dtype=bool)
        test[fold] = True
        for j in range(Y.shape[1]):
            model = fit_gp(
                X[~test],
                Y[~test, j],
                lengthscale=lengthscale,
                lam=RIDGE_LAM,
                B=1.0,
                R=0.0,
                c=c[~test],
            )
            mu, _ = posterior(model, X[test], c[test])
            sq_errors.append((mu - Y[test, j]) ** 2)
    return float(np.sqrt(np.concatenate(sq_errors).mean()))


def select_lengthscale(
    X: np.ndarray, c: np.ndarray, Y: np.ndarray
) -> tuple[float, float, float, dict[str, float]]:
    """Select the frozen lengthscale.

    Median pairwise distance on the calibration x-points times each
    multiplier in {0.5, 1.0, 2.0}; the multiplier with the lowest 5-fold
    kernel-ridge RMSE (lam 1e-6) wins. Folds come from the registered
    split-seed permutation.

    Args:
        X: Unit design points, shape (512, x_dim).
        c: Unit coefficients, shape (512,).
        Y: Standardized margins, shape (512, m).

    Returns:
        Tuple (lengthscale, multiplier, median_distance, rmse_by_multiplier).
    """
    med = median_pairwise_distance(X)
    perm = np.random.default_rng(SPLIT_SEED).permutation(X.shape[0])
    folds = np.array_split(perm, N_CV_FOLDS)
    rmse: dict[str, float] = {}
    for mult in LENGTHSCALE_MULTIPLIERS:
        rmse[str(mult)] = _cv_rmse(X, c, Y, med * mult, folds)
        logger.info(
            "lengthscale CV: multiplier %.1f -> ls %.4f, RMSE %.4f",
            mult,
            med * mult,
            rmse[str(mult)],
        )
    best = min(LENGTHSCALE_MULTIPLIERS, key=lambda mu: rmse[str(mu)])
    return med * best, float(best), med, rmse


def _interpolant_norm(
    X: np.ndarray,
    y: np.ndarray,
    lengthscale: float,
    c: np.ndarray | None,
    lam: float,
) -> float:
    """RKHS norm of a kernel-ridge fit of (X, y)."""
    model = fit_gp(X, y, lengthscale=lengthscale, lam=lam, B=1.0, R=0.0, c=c)
    K = rbf(X, X, lengthscale)
    if c is not None:
        K = K * (1.0 + np.outer(c, c))
    return float(np.sqrt(max(model.w @ (K @ model.w), 0.0)))


def norm_bounds(
    X: np.ndarray,
    c: np.ndarray,
    Y: np.ndarray,
    lengthscale: float,
    *,
    lam: float = RIDGE_LAM,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen norm bounds B_j and B_slice_j.

    B_j: inflated (2x) RKHS norm of the joint kernel-ridge fit on all 512
    calibration points. B_slice_j: inflated maximum x-only fit norm over
    the 8 c-anchor groups of the Sobol block.

    Args:
        X: Unit design points, shape (512, x_dim); first 256 rows Sobol.
        c: Unit coefficients, shape (512,).
        Y: Standardized margins, shape (512, m).
        lengthscale: Frozen RBF lengthscale on X.
        lam: norm-fit regularization. Deterministic environments use
            1e-6; stochastic environments use R(lowest knob)^2.

    Returns:
        Tuple (B, B_slice), each of shape (m,).
    """
    m = Y.shape[1]
    B = np.empty(m)
    for j in range(m):
        B[j] = NORM_INFLATION * _interpolant_norm(X, Y[:, j], lengthscale, c, lam)
    anchors = np.linspace(0.0, 1.0, N_SLICE_ANCHORS)
    sobol_idx = np.arange(N_SOBOL)
    nearest = np.argmin(np.abs(c[sobol_idx][:, None] - anchors[None, :]), axis=1)
    B_slice = np.zeros(m)
    for a in range(N_SLICE_ANCHORS):
        grp = sobol_idx[nearest == a]
        if grp.size == 0:
            continue
        for j in range(m):
            B_slice[j] = max(
                B_slice[j],
                NORM_INFLATION
                * _interpolant_norm(X[grp], Y[grp, j], lengthscale, None, lam),
            )
    return B, B_slice


def holdout_r2(
    X: np.ndarray, c: np.ndarray, Y: np.ndarray, lengthscale: float
) -> np.ndarray:
    """Held-out 20% calibration R^2 per channel (acceptance, step 6).

    The split-seed permutation defines the split: the first 80%
    trains the frozen-lengthscale kernel ridge (lam 1e-6), the last 20% is
    held out.

    Args:
        X: Unit design points, shape (512, x_dim).
        c: Unit coefficients, shape (512,).
        Y: Standardized margins, shape (512, m).
        lengthscale: Frozen RBF lengthscale on X.

    Returns:
        R^2 per channel, shape (m,).
    """
    n = X.shape[0]
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_test = int(round(HOLDOUT_FRACTION * n))
    test, train = perm[n - n_test :], perm[: n - n_test]
    r2 = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        model = fit_gp(
            X[train],
            Y[train, j],
            lengthscale=lengthscale,
            lam=RIDGE_LAM,
            B=1.0,
            R=0.0,
            c=c[train],
        )
        mu, _ = posterior(model, X[test], c[test])
        resid = Y[test, j] - mu
        total = Y[test, j] - Y[test, j].mean()
        r2[j] = 1.0 - float(resid @ resid) / float(total @ total)
    return r2


def calibrate(
    experiment: str,
    raw_oracle: RawOracle,
    x_dim: int,
    m: int,
    *,
    crash_margin: float = 3.0,
    R: float = 0.0,
    clip: bool = True,
) -> FrozenCalibration:
    """Run the full deterministic-environment calibration recipe.

    Args:
        experiment: Experiment name recorded in the frozen constants.
        raw_oracle: Raw-margin oracle on unit coordinates; None on crash.
            Stochastic environments must bind it to the lowest noise knob.
        x_dim: Design-space dimension.
        m: Number of margin channels.
        crash_margin: Standardized crash margin.
        R: Frozen noise scale (0 for deterministic environments; lam is
            then 1e-6, else R^2 per the noise-matched convention).
        clip: Whether to clip standardized margins (False for window
            channels per amendment P2).

    Returns:
        FrozenCalibration with every calibration constant.
    """
    X, c = calibration_points(x_dim)
    Y_raw, crashed = evaluate_raw(raw_oracle, X, c, m)
    logger.info(
        "calibration: %d points evaluated, %d crashed", X.shape[0], crashed.sum()
    )
    median, scale = fit_scalers(Y_raw, crashed)
    Y = standardize(Y_raw, crashed, scale, crash_margin, clip=clip)
    lengthscale, mult, med_dist, rmse = select_lengthscale(X, c, Y)
    # Amendment P6: estimate the norm of the mean function rather than
    # interpolating stochastic calibration noise. Deterministic
    # environments retain the 1e-6 interpolation branch.
    norm_lam = RIDGE_LAM if R == 0.0 else R * R
    B, B_slice = norm_bounds(X, c, Y, lengthscale, lam=norm_lam)
    r2 = holdout_r2(X, c, Y, lengthscale)
    r2_ok = bool(np.all(r2 >= R2_MIN))
    logger.info(
        "calibration: lengthscale=%.4f (mult %.1f), B=%s, B_slice=%s, R^2=%s",
        lengthscale,
        mult,
        np.round(B, 4),
        np.round(B_slice, 4),
        np.round(r2, 4),
    )
    return FrozenCalibration(
        experiment=experiment,
        x_dim=x_dim,
        m=m,
        scaler_median=tuple(float(v) for v in median),
        scaler_scale=tuple(float(v) for v in scale),
        crash_margin=float(crash_margin),
        n_crashed=int(crashed.sum()),
        median_pairwise_distance_x=med_dist,
        lengthscale_multiplier=mult,
        lengthscale=float(lengthscale),
        cv_rmse_by_multiplier=rmse,
        B=tuple(float(v) for v in B),
        B_slice=tuple(float(v) for v in B_slice),
        R=float(R),
        lam=float(RIDGE_LAM if R == 0.0 else R * R),
        r2_holdout=tuple(float(v) for v in r2),
        r2_ok=r2_ok,
        seeds={"sobol": SOBOL_SEED, "uniform": UNIFORM_SEED, "split": SPLIT_SEED},
        clip=clip,
    )


def save_calibration(cal: FrozenCalibration, path: str | Path) -> None:
    """Write frozen calibration constants to JSON.

    Args:
        cal: Frozen calibration constants.
        path: Destination JSON path.
    """
    Path(path).write_text(json.dumps({"calibration": asdict(cal)}, indent=2))


def load_calibration(path: str | Path) -> FrozenCalibration:
    """Load frozen calibration constants from JSON.

    Args:
        path: Source JSON path (the ``calibration`` key holds the constants).

    Returns:
        FrozenCalibration.
    """
    payload = json.loads(Path(path).read_text())["calibration"]
    payload["scaler_median"] = tuple(payload["scaler_median"])
    payload["scaler_scale"] = tuple(payload["scaler_scale"])
    payload["B"] = tuple(payload["B"])
    payload["B_slice"] = tuple(payload["B_slice"])
    payload["r2_holdout"] = tuple(payload["r2_holdout"])
    return FrozenCalibration(**payload)
