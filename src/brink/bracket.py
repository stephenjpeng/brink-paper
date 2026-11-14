"""Lower and upper bounds on the operating limit theta = sup{c : F(c) nonempty}.

On the event that the running confidence bounds contain every constraint
(probability at least 1 - delta; Theorem 1):

  * Lower bound (feasible design): any point with max_j u_j(x, c) <= 0 is
    feasible at c, so L = the largest such c over the scanned candidates and
    theta >= L. Failing to find such a design proves nothing.
  * Upper bound (exclusion): at a stress c, if every cell of a cover of the
    design box with center p and radius rho has some j with
    l_j(p, c) - D_j rho > 0 (strict), every design violates a constraint at
    c, so F(c) is empty. With the upper-window constraint, or
    when feasible sets shrink with stress, exclusion extends to higher
    stresses, so U = the running minimum of excluded stresses.

The adaptive test refines unresolved cells to a maximum depth and cell count,
and can add the derivative-based (second-order) test of Appendix A.4.

Lipschitz bound: for f in the RKHS of an RBF kernel with lengthscale h and
||f|| <= Bf, |f(x) - f(y)| <= Bf ||x - y|| / h, and the posterior standard
deviation has the same constant 1/h. Hence l = mu - b sd is Lipschitz in x
with constant (||mu|| + b) L_kappa / h, where ||mu||^2 = w^T K w and
L_kappa = sqrt(1 + c_max^2) for the affine joint kernel k(x, x')(1 + c c').
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from brink.gp import GPModel, fit_gp, posterior
from brink.kernels import rbf

MODE_AFFINE = "affine"  # joint kernel rbf(x,x') * (1 + c c'); model.c is set
MODE_RBF = "rbf"  # concatenated inputs z = (x, c); model.c is None
GRAD_CHUNK = 2048  # query rows per gradient block: (rows x n x d) transients


@dataclass
class JointBands:
    """Per-constraint joint model with a fixed confidence radius.

    Attributes:
        models: One fitted GPModel per constraint.
        betas: Confidence radii beta_j^{1/2} (fixed per report).
        mode: MODE_AFFINE or MODE_RBF.
        c_max: Upper end of the coefficient range (affine Lipschitz factor).
        offsets: Additive band widening eta_j per constraint (model mismatch,
            manuscript Appendix A.1): lcb = mu - beta sd - eta and
            ucb = mu + beta sd + eta. None means zero. A constant does not
            change the Lipschitz constant of lcb, so ``lipschitz`` holds as is.
    """

    models: list[GPModel]
    betas: list[float]
    mode: str
    c_max: float = 1.0
    offsets: list[float] | None = None

    def _offset(self, j: int) -> float:
        return 0.0 if self.offsets is None else float(self.offsets[j])

    def _mu_sd(self, j: int, X: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == MODE_AFFINE:
            return posterior(self.models[j], X, float(c))
        Z = np.concatenate([X, np.full((X.shape[0], 1), float(c))], axis=1)
        return posterior(self.models[j], Z, None)

    def subset(self, channels: list[int]) -> JointBands:
        """The same bands restricted to the listed constraint channels.

        A cover test on a subset stays sound on E, whose simultaneous event
        covers every channel. On a two-sided window family, the death channel
        g_hi = c - c_off(x) is nondecreasing in c, so a slice it certifies
        stays empty at every higher c, on or off the grid, without the nesting
        assumption.
        """
        offsets = self.offsets
        return JointBands(
            models=[self.models[j] for j in channels],
            betas=[self.betas[j] for j in channels],
            mode=self.mode,
            c_max=self.c_max,
            offsets=None if offsets is None else [offsets[j] for j in channels],
        )

    def bands(self, X: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Stacked (lcb, ucb), each of shape (m, n), at one slice."""
        lcb, ucb = [], []
        for j in range(len(self.models)):
            mu, sd = self._mu_sd(j, X, c)
            lcb.append(mu - self.betas[j] * sd - self._offset(j))
            ucb.append(mu + self.betas[j] * sd + self._offset(j))
        return np.stack(lcb), np.stack(ucb)

    def sd_max(self, X: np.ndarray, c_arr: np.ndarray) -> np.ndarray:
        """Max posterior sd across constraints at paired (x, c) points."""
        out = np.zeros(X.shape[0])
        for j in range(len(self.models)):
            if self.mode == MODE_AFFINE:
                _, sd = posterior(self.models[j], X, c_arr)
            else:
                Z = np.concatenate([X, c_arr[:, None]], axis=1)
                _, sd = posterior(self.models[j], Z, None)
            np.maximum(out, sd, out=out)
        return out

    def undetermined(
        self, X: np.ndarray, c_arr: np.ndarray, margin: np.ndarray | None = None
    ) -> np.ndarray:
        """Not provably infeasible: all lcb_j <= margin_j at paired (x, c).

        The default margin 0 is the plain test. A positive margin (the cover
        certificate's Lipschitz clearance) keeps points that are past 0 but
        still block the strict cover certificate.
        """
        margin = np.zeros(len(self.models)) if margin is None else margin
        ok = np.ones(X.shape[0], dtype=bool)
        for j in range(len(self.models)):
            if self.mode == MODE_AFFINE:
                mu, sd = posterior(self.models[j], X, c_arr)
            else:
                Z = np.concatenate([X, c_arr[:, None]], axis=1)
                mu, sd = posterior(self.models[j], Z, None)
            ok &= (mu - self.betas[j] * sd - self._offset(j)) <= margin[j]
        return ok

    def grad_bands(self, X: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form grad mu_j and gradient posterior sd s_grad,j at one slice.

        s_grad,j(x)^2 is the largest eigenvalue of the posterior covariance of
        grad_x g_j, (kappa_c^2 / l^2) I - J^T (K + lam I)^-1 J with
        J_ia = d k((x, c), z_i) / d x_a and kappa_c^2 = k((x, c), (x, c))
        (Appendix A.4).

        Returns:
            (grad_mu, s_grad) of shapes (m, q, d) and (m, q).
        """
        q, d = X.shape
        grad = np.empty((len(self.models), q, d))
        s_grad = np.empty((len(self.models), q))
        for j, model in enumerate(self.models):
            ls2 = model.lengthscale**2
            prior = (1.0 + c * c if self.mode == MODE_AFFINE else 1.0) / ls2
            for s in range(0, q, GRAD_CHUNK):
                Xs = X[s : s + GRAD_CHUNK]
                if self.mode == MODE_AFFINE:
                    Kq = rbf(Xs, model.X, model.lengthscale) * (1.0 + c * model.c)
                else:
                    Zs = np.column_stack([Xs, np.full(len(Xs), float(c))])
                    Kq = rbf(Zs, model.X, model.lengthscale)
                J = Kq[:, :, None] * (model.X[None, :, :d] - Xs[:, None]) / ls2
                grad[j, s : s + len(Xs)] = np.einsum("bnd,n->bd", J, model.w)
                A = solve_triangular(
                    model.L, J.transpose(1, 0, 2).reshape(model.n, -1), lower=True
                ).reshape(model.n, len(Xs), d)
                cov = prior * np.eye(d) - np.einsum("nba,nbe->bae", A, A)
                top = np.linalg.eigvalsh(cov)[:, -1]
                s_grad[j, s : s + len(Xs)] = np.sqrt(np.maximum(top, 0.0))
        return grad, s_grad

    def lipschitz(self, j: int) -> float:
        """Sound Lipschitz constant of lcb_j in x (docstring derivation)."""
        model = self.models[j]
        if model.n == 0:
            mu_norm = 0.0
        else:
            K = rbf(model.X, model.X, model.lengthscale)
            if model.c is not None:
                K = K * (1.0 + np.outer(model.c, model.c))
            mu_norm = float(np.sqrt(max(model.w @ (K @ model.w), 0.0)))
        factor = np.sqrt(1.0 + self.c_max**2) if self.mode == MODE_AFFINE else 1.0
        return (mu_norm + self.betas[j]) * factor / model.lengthscale


def fit_joint_bands(
    X: np.ndarray,
    c: np.ndarray,
    Y: np.ndarray,
    *,
    mode: str,
    lengthscale: float,
    lam: float,
    B: np.ndarray,
    R: float,
    betas: list[float],
    c_max: float = 1.0,
) -> JointBands:
    """Fit one joint GP per constraint in the requested mode."""
    models = []
    for j in range(Y.shape[1]):
        if mode == MODE_AFFINE:
            models.append(
                fit_gp(
                    X,
                    Y[:, j],
                    lengthscale=lengthscale,
                    lam=lam,
                    B=float(B[j]),
                    R=R,
                    c=c,
                )
            )
        else:
            Z = np.concatenate([X, c[:, None]], axis=1)
            models.append(
                fit_gp(Z, Y[:, j], lengthscale=lengthscale, lam=lam, B=float(B[j]), R=R)
            )
    return JointBands(models=models, betas=betas, mode=mode, c_max=c_max)


def midpoint_cover(n_per_axis: int, x_dim: int) -> tuple[np.ndarray, float]:
    """Midpoint grid cover of [0,1]^d and its covering radius rho."""
    axes = [(np.arange(n_per_axis) + 0.5) / n_per_axis] * x_dim
    mesh = np.meshgrid(*axes, indexing="ij")
    cover = np.stack([m.ravel() for m in mesh], axis=1)
    rho = 0.5 * np.sqrt(x_dim) / n_per_axis
    return cover, rho


@dataclass
class ExistenceBracket:
    """One report's certified bracket for theta_exists.

    Attributes:
        lower: Witness lower endpoint L_N (c_min with flag when no witness).
        upper: Certificate upper endpoint U_N (c_max with flag when none).
        has_witness: Whether any certified-feasible witness was found.
        has_certificate: Whether any slice certified empty.
        witness_x: The witness design point (or None).
    """

    lower: float
    upper: float
    has_witness: bool
    has_certificate: bool
    witness_x: np.ndarray | None

    @property
    def width(self) -> float:
        """Bracket width U_N - L_N."""
        return self.upper - self.lower


def witness_lower_bound(
    bands: JointBands, candidates: np.ndarray, c_grid: np.ndarray
) -> tuple[float, bool, np.ndarray | None]:
    """Largest grid c with a certified-feasible candidate (ucb_j <= 0 all j).

    Scans slices from the top; sound on E for any candidate set.
    """
    for c in c_grid[::-1]:
        _, ucb = bands.bands(candidates, float(c))
        feas = np.all(ucb <= 0.0, axis=0)
        if feas.any():
            return float(c), True, candidates[int(np.argmax(feas))]
    return float(c_grid[0]), False, None


def _refined_cell_points(
    centers: np.ndarray, cell_half_width: float, factor: int
) -> np.ndarray:
    """Midpoint subgrid of factor^d points inside each failing cell."""
    d = centers.shape[1]
    offsets_1d = (np.arange(factor) + 0.5) / factor * 2.0 - 1.0  # in (-1, 1)
    mesh = np.meshgrid(*([offsets_1d] * d), indexing="ij")
    offsets = np.stack([m.ravel() for m in mesh], axis=1) * cell_half_width
    pts = centers[:, None, :] + offsets[None, :, :]
    return pts.reshape(-1, d)


def _check_slice(bands: JointBands, c: float) -> None:
    """Raise when the affine Lipschitz factor sqrt(1 + c_max^2) does not bound c."""
    if bands.mode == MODE_AFFINE and abs(c) > bands.c_max:
        raise ValueError(f"slice c = {c} outside |c| <= c_max = {bands.c_max}")


def certificate_upper_bound(
    bands: JointBands,
    c_grid: np.ndarray,
    cover: np.ndarray,
    rho: float,
    *,
    stop_below: float | None = None,
    refine_factor: int = 8,
    max_refine_cells: int = 4000,
    max_depth: int = 0,
    max_cells: int = 8_000_000,
    second_order: bool = False,
    time_cap_s: float | None = None,
) -> tuple[float, bool]:
    """Smallest grid slice certified empty by the strict cover test.

    Scans from the top while slices certify (valid under the registered
    nested-family property: emptiness extends upward, so the certified set
    is a top interval and scanning may stop at the first failure or at
    stop_below). STRICT inequality per the corrected Theorem 2(b) rule.

    max_depth > 0 replaces the two-level cell test by the adaptive test
    :func:`_slice_certifies_adaptive` with that depth and per-slice cell
    budget max_cells (S3 amendment S3-A3: the no-witness end scan descends
    from c_max); second_order adds the second-order clearance to it.
    time_cap_s stops the descent once that many seconds have elapsed and
    returns the slices certified so far, which is a valid but not the
    smallest bound; the caller reads a cap hit from its own clock.

    Two-level sound refinement (the branch-and-bound-lite route the proofs
    contemplate): coarse cells failing the clearance test are re-tested on
    a factor^d midpoint subgrid whose covering radius is rho / factor, so
    the Lipschitz term shrinks by the factor exactly where it binds. A
    slice certifies when every coarse point clears at radius rho or its
    refined subgrid clears at radius rho / factor. Refinement is skipped
    (slice fails) when more than max_refine_cells cells need it: near-theta
    slices are then simply not certified, which is sound.
    """
    if second_order and max_depth <= 0:
        raise ValueError("second-order clearance needs the adaptive cover (max_depth)")
    m = len(bands.models)
    lip = np.array([bands.lipschitz(j) for j in range(m)])
    cell_half_width = rho / np.sqrt(cover.shape[1])  # midpoint-grid half step
    certified_until = None
    t0 = time.perf_counter()
    for c in c_grid[::-1]:
        if stop_below is not None and c < stop_below:
            break
        if time_cap_s is not None and time.perf_counter() - t0 > time_cap_s:
            break
        if max_depth > 0:
            slice_ok = _slice_certifies_adaptive(
                bands, float(c), cover, rho, lip, max_depth, max_cells, second_order
            )
            if slice_ok:
                certified_until = float(c)
                continue
            break
        _check_slice(bands, float(c))
        lcb, _ = bands.bands(cover, float(c))
        coarse_ok = np.any(lcb - lip[:, None] * rho > 0.0, axis=0)  # (P,)
        failing = np.flatnonzero(~coarse_ok)
        slice_ok = failing.size == 0
        if not slice_ok and failing.size <= max_refine_cells and refine_factor > 1:
            fine = _refined_cell_points(cover[failing], cell_half_width, refine_factor)
            lcb_f, _ = bands.bands(fine, float(c))
            fine_ok = np.any(lcb_f - lip[:, None] * (rho / refine_factor) > 0.0, axis=0)
            slice_ok = bool(np.all(fine_ok))
        if slice_ok:
            certified_until = float(c)
        else:
            break
    if certified_until is None:
        return float(c_grid[-1]), False
    return certified_until, True


def _slice_certifies(
    bands: JointBands,
    c: float,
    cover: np.ndarray,
    rho: float,
    lip: np.ndarray,
    refine_factor: int,
    max_refine_cells: int,
    batch_cells: int = 64,
) -> bool:
    """The strict two-level cover test of one slice, with early exit.

    Same decision as the per-slice test in :func:`certificate_upper_bound`.
    Failing coarse cells are refined worst-first in batches, and the slice
    fails at the first refined point that does not clear, so slices that
    clearly fail (below theta, nearly every cell) cost about one batch.
    """
    _check_slice(bands, float(c))
    cell_half_width = rho / np.sqrt(cover.shape[1])
    lcb, _ = bands.bands(cover, float(c))
    margin = lcb - lip[:, None] * rho
    failing = np.flatnonzero(~np.any(margin > 0.0, axis=0))  # test
    if failing.size == 0:
        return True
    if failing.size > max_refine_cells or refine_factor <= 1:
        return False
    failing = failing[np.argsort(np.max(margin[:, failing], axis=0))]
    for i in range(0, failing.size, batch_cells):
        fine = _refined_cell_points(
            cover[failing[i : i + batch_cells]], cell_half_width, refine_factor
        )
        lcb_f, _ = bands.bands(fine, float(c))
        fine_clear = np.any(lcb_f - lip[:, None] * (rho / refine_factor) > 0.0, axis=0)
        if not np.all(fine_clear):
            return False
    return True


def _slice_certifies_adaptive(
    bands: JointBands,
    c: float,
    cover: np.ndarray,
    rho: float,
    lip: np.ndarray,
    max_depth: int,
    max_cells: int,
    second_order: bool = False,
) -> bool:
    """Branch-and-bound form of the strict cover test at one slice.

    Cells start as the midpoint cover (half-width rho / sqrt(d), radius rho). A
    cell clears when some j has lcb_j(center) > L_j * radius (STRICT); every
    other cell splits into 2^d children of half the width, down to max_depth
    levels. The slice certifies when no cell is left. Sound on E by the same
    Lipschitz argument as the two-level test: every x lies in a cleared cell.
    Depth 3 reaches exactly the factor-8 subgrid, so with enough cells this test
    certifies every slice the two-level test does. It fails at once when a
    center has every lcb_j <= 0 (no cell containing it can clear), and when the
    next level would take the total cell evaluations above max_cells.

    second_order = True also clears a cell when some j has
    lcb_j(z) - r G_j(z) - r^2 Q_j > 0 (OR with the first-order test), with
    G_j = ||grad mu_j(z)|| + b_j s_grad,j(z) and Q_j = sqrt(3) L_j / (2 l)
    (Appendix A.4).
    """
    _check_slice(bands, float(c))
    d = cover.shape[1]
    if second_order:
        ls = np.array([mdl.lengthscale for mdl in bands.models])
        Q = np.sqrt(3.0) * lip / (2.0 * ls)
        b = np.asarray(bands.betas, dtype=float)
    signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * d), indexing="ij")).reshape(d, -1).T
    centers, radius = cover, rho
    half = rho / np.sqrt(d)
    n_eval = 0
    for level in range(max_depth + 1):
        n_eval += len(centers)
        lcb, _ = bands.bands(centers, float(c))
        if np.any(np.all(lcb <= 0.0, axis=0)):
            return False
        open_ = ~np.any(lcb - lip[:, None] * radius > 0.0, axis=0)
        if second_order and open_.any():
            grad, s_grad = bands.grad_bands(centers[open_], float(c))
            G = np.linalg.norm(grad, axis=2) + b[:, None] * s_grad
            margin2 = lcb[:, open_] - radius * G - radius**2 * Q[:, None]
            open_[open_] = ~np.any(margin2 > 0.0, axis=0)
        n_open = int(open_.sum())
        if n_open == 0:
            return True
        if level == max_depth or n_eval + n_open * len(signs) > max_cells:
            return False
        half, radius = half / 2, radius / 2
        centers = (centers[open_][:, None, :] + half * signs[None]).reshape(-1, d)
    return False


def smallest_certified_slice(
    bands: JointBands,
    c_grid: np.ndarray,
    cover: np.ndarray,
    rho: float,
    *,
    stop_below: float | None = None,
    refine_factor: int = 8,
    max_refine_cells: int = 4000,
    max_depth: int = 0,
    max_cells: int = 8_000_000,
    second_order: bool = False,
    time_cap_s: float | None = None,
) -> tuple[float, bool]:
    """Smallest grid slice certified empty: the derivation's rule under nesting.

    Under the nested-family property (emptiness extends upward),
    any certified slice bounds theta_exists from above on E, so U_N is the
    smallest certified slice. With the upper-window constraint alone
    (``JointBands.subset([1])``) no nesting is needed (Section 5). Scans upward
    from stop_below (the witness endpoint) or c_min
    and returns the first certified slice. On the same bands it is never
    above the :func:`certificate_upper_bound` result, which stops at the
    first failing slice from the top. Both rules need nesting:
    without it, the derivation also needs a Lipschitz-in-c clearance for the
    gaps between grid slices, which neither implements.

    max_depth = 0 keeps the two-level slice test; max_depth > 0
    uses the adaptive test :func:`_slice_certifies_adaptive` with that depth
    and the per-slice cell budget max_cells (refine_factor is then unused);
    second_order adds the second-order clearance to that adaptive test.
    time_cap_s stops the scan once that many seconds have elapsed and returns
    the trivial endpoint (the S3 capped end-of-run scan); the caller reads a
    cap hit from its own clock.

    Returns:
        (U_N, has_certificate); (c_max, False) when no slice certifies.
    """
    lip = np.array([bands.lipschitz(j) for j in range(len(bands.models))])
    t0 = time.perf_counter()
    for c in c_grid:
        if stop_below is not None and c < stop_below:
            continue
        if time_cap_s is not None and time.perf_counter() - t0 > time_cap_s:
            return float(c_grid[-1]), False
        if max_depth > 0:
            ok = _slice_certifies_adaptive(
                bands, float(c), cover, rho, lip, max_depth, max_cells, second_order
            )
        else:
            ok = _slice_certifies(
                bands, float(c), cover, rho, lip, refine_factor, max_refine_cells
            )
        if ok:
            return float(c), True
    return float(c_grid[-1]), False


SCAN_TOP_DOWN = "top-down"  # descend from c_max (needs nesting)
SCAN_SMALLEST = "smallest"  # derivation rule under nesting (smallest slice)
SCAN_OFF = "off"  # no certificate; upper endpoint stays c_max


def compute_bracket(
    bands: JointBands,
    candidates: np.ndarray,
    c_grid: np.ndarray,
    cover: np.ndarray,
    rho: float,
    refine_factor: int = 8,
    scan: str = SCAN_TOP_DOWN,
    max_depth: int = 0,
    max_cells: int = 8_000_000,
    second_order: bool = False,
) -> ExistenceBracket:
    """Witness lower bound plus certificate upper bound at one report.

    max_depth > 0 switches the smallest scan to the adaptive cover test;
    the top-down scan keeps its two-level form.
    second_order adds the second-order clearance to the adaptive test
    (default off). scan = SCAN_OFF computes the witness side only and keeps
    upper = c_max.
    """
    lower, has_w, wx = witness_lower_bound(bands, candidates, c_grid)
    if scan == SCAN_OFF:
        return ExistenceBracket(
            lower=lower if has_w else float(c_grid[0]),
            upper=float(c_grid[-1]),
            has_witness=has_w,
            has_certificate=False,
            witness_x=wx,
        )
    if max_depth > 0 and scan != SCAN_SMALLEST:
        raise ValueError(f"adaptive cover needs scan={SCAN_SMALLEST!r}, got {scan!r}")
    if second_order and max_depth <= 0:
        raise ValueError("second-order clearance needs the adaptive cover (max_depth)")
    certificate = {
        SCAN_TOP_DOWN: certificate_upper_bound,
        SCAN_SMALLEST: smallest_certified_slice,
    }[scan]
    adaptive = (
        {"max_depth": max_depth, "max_cells": max_cells, "second_order": second_order}
        if max_depth
        else {}
    )
    upper, has_c = certificate(
        bands,
        c_grid,
        cover,
        rho,
        stop_below=lower if has_w else None,
        refine_factor=refine_factor,
        **adaptive,
    )
    if has_w and upper < lower:  # off-event symptom: keep the sound order
        upper = float(c_grid[-1])
        has_c = False
    return ExistenceBracket(
        lower=lower if has_w else float(c_grid[0]),
        upper=upper,
        has_witness=has_w,
        has_certificate=has_c,
        witness_x=wx,
    )
