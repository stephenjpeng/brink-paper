"""Tests for the certified existence bracket (witness + cover certificate)."""

import numpy as np

from brink.bracket import (
    MODE_RBF,
    SCAN_SMALLEST,
    certificate_upper_bound,
    compute_bracket,
    fit_joint_bands,
    midpoint_cover,
    smallest_certified_slice,
)


class _ToySpec:
    x_dim = 2
    m = 1
    c_min = 0.0
    c_max = 1.0


class ToyEnv:
    """g_1(x, c) = c - 0.6 (constant in x): theta_exists = 0.6 exactly."""

    spec = _ToySpec

    def g(self, X, c):
        return np.full((1, np.atleast_2d(X).shape[0]), c - 0.6)


def _dense_bands(n=200, beta_mult=1.0):
    env = ToyEnv()
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n, 2))
    c = rng.uniform(size=n)
    Y = np.stack([env.g(X[i : i + 1], float(c[i]))[:, 0] for i in range(n)])
    bands = fit_joint_bands(
        X,
        c,
        Y,
        mode=MODE_RBF,
        lengthscale=0.5,
        lam=1e-6,
        B=np.array([2.0]),
        R=0.0,
        betas=[2.0 * beta_mult],
        c_max=1.0,
    )
    return bands


def test_midpoint_cover_radius():
    cover, rho = midpoint_cover(9, 2)
    pts = np.random.default_rng(1).uniform(size=(500, 2))
    d = np.min(np.linalg.norm(pts[:, None, :] - cover[None, :, :], axis=2), axis=1)
    assert np.all(d <= rho + 1e-12)


def test_bracket_contains_toy_theta_and_tightens_with_cover():
    c_grid = np.linspace(0.0, 1.0, 101)
    cands = np.random.default_rng(2).uniform(size=(256, 2))
    bands = _dense_bands(200)
    widths = []
    for n_cover in (11, 41):  # finer cover shrinks the L * rho term
        cover, rho = midpoint_cover(n_cover, 2)
        br = compute_bracket(bands, cands, c_grid, cover, rho)
        assert br.lower <= 0.6 <= br.upper
        assert br.lower <= br.upper
        widths.append(br.width)
    assert widths[1] < widths[0]
    cover, rho = midpoint_cover(41, 2)
    br = compute_bracket(bands, cands, c_grid, cover, rho)
    assert br.has_witness and br.has_certificate
    # The witness is truly feasible at the lower endpoint (soundness).
    env = ToyEnv()
    assert np.all(env.g(br.witness_x[None, :], br.lower) <= 0.0)


def test_no_witness_flags_lower_at_c_min():
    c_grid = np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(11, 2)
    cands = np.random.default_rng(3).uniform(size=(64, 2))
    # Huge beta: nothing certifies feasible, nothing certifies empty.
    bands = _dense_bands(40, beta_mult=100.0)
    br = compute_bracket(bands, cands, c_grid, cover, rho)
    assert not br.has_witness and not br.has_certificate
    assert br.lower == 0.0 and br.upper == 1.0


def test_lipschitz_bound_is_sound_on_samples():
    bands = _dense_bands(100)
    lip = bands.lipschitz(0)
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(200, 2))
    Y = X + rng.normal(scale=0.01, size=X.shape)
    Y = np.clip(Y, 0.0, 1.0)
    lcb_x, _ = bands.bands(X, 0.5)
    lcb_y, _ = bands.bands(Y, 0.5)
    slopes = np.abs(lcb_x - lcb_y)[0] / np.maximum(np.linalg.norm(X - Y, axis=1), 1e-12)
    assert np.max(slopes) <= lip + 1e-8


def test_bracket_endpoints_unchanged_under_chunking(monkeypatch):
    """Chunked posterior queries leave every bracket field identical (P2)."""
    import brink.gp as gp

    c_grid = np.linspace(0.0, 1.0, 101)
    cover, rho = midpoint_cover(41, 2)
    cands = np.random.default_rng(2).uniform(size=(256, 2))
    bands = _dense_bands(200)
    monkeypatch.setattr(gp, "POSTERIOR_CHUNK", 10**9)
    full = compute_bracket(bands, cands, c_grid, cover, rho)
    monkeypatch.setattr(gp, "POSTERIOR_CHUNK", 7)
    chunked = compute_bracket(bands, cands, c_grid, cover, rho)
    assert chunked.lower == full.lower
    assert chunked.upper == full.upper
    assert chunked.has_witness == full.has_witness
    assert chunked.has_certificate == full.has_certificate


def test_undetermined_margin_keeps_a_superset():
    bands = _dense_bands()
    rng = np.random.default_rng(1)
    X, c = rng.uniform(size=(500, 2)), rng.uniform(size=500)
    plain = bands.undetermined(X, c)
    assert np.array_equal(plain, bands.undetermined(X, c, np.zeros(1)))
    wide = bands.undetermined(X, c, np.array([0.5]))
    assert np.all(wide[plain]) and wide.sum() > plain.sum()


def _gap_bands():
    """Nested toy whose clearance shrinks near c = 1: g = 8 (c - 0.6)(1.05 - c).

    F(c) is empty for every c > 0.6 (theta_exists = 0.6), but the top slices
    clear by less, so they fail the cover test while 0.68-0.96 certify.
    """
    rng = np.random.default_rng(0)
    X, c = rng.uniform(size=(300, 2)), rng.uniform(size=300)
    Y = (8.0 * (c - 0.6) * (1.05 - c))[:, None]
    return fit_joint_bands(
        X,
        c,
        Y,
        mode=MODE_RBF,
        lengthscale=0.5,
        lam=1e-6,
        B=np.array([2.0]),
        R=0.0,
        betas=[1.0],
        c_max=1.0,
    )


def _brute_force_smallest(bands, c_grid, cover, rho, stop_below=None):
    """Per-slice test applied to every slice; smallest certified."""
    hits = [
        c
        for c in c_grid
        if (stop_below is None or c >= stop_below)
        and certificate_upper_bound(bands, np.array([c]), cover, rho)[1]
    ]
    return (float(min(hits)), True) if hits else (float(c_grid[-1]), False)


def test_smallest_scan_recovers_certificate_below_failing_top_slices():
    bands, grid = _gap_bands(), np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(17, 2)
    assert certificate_upper_bound(bands, grid, cover, rho) == (1.0, False)
    upper, ok = smallest_certified_slice(bands, grid, cover, rho)
    assert ok and abs(upper - 0.68) < 1e-12 and upper >= 0.6  # sound: theta = 0.6


def test_smallest_scan_is_exact_and_never_above_top_down():
    grid = np.linspace(0.0, 1.0, 51)
    for bands, n_axis in ((_gap_bands(), 17), (_dense_bands(), 9)):
        cover, rho = midpoint_cover(n_axis, 2)
        for stop in (None, 0.7):
            got = smallest_certified_slice(bands, grid, cover, rho, stop_below=stop)
            assert got == _brute_force_smallest(bands, grid, cover, rho, stop)
            top = certificate_upper_bound(bands, grid, cover, rho, stop_below=stop)
            assert got[0] <= top[0]


def test_compute_bracket_scan_switch():
    bands, grid = _gap_bands(), np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(17, 2)
    cands = np.random.default_rng(3).uniform(size=(64, 2))
    default = compute_bracket(bands, cands, grid, cover, rho)
    smallest = compute_bracket(bands, cands, grid, cover, rho, scan=SCAN_SMALLEST)
    assert not default.has_certificate and smallest.has_certificate
    assert smallest.upper < default.upper


def test_slice_test_matches_registered_per_slice_decision():
    from brink.bracket import _slice_certifies

    grid = np.linspace(0.0, 1.0, 51)
    for bands, n_axis in ((_gap_bands(), 17), (_dense_bands(), 9)):
        cover, rho = midpoint_cover(n_axis, 2)
        lip = np.array([bands.lipschitz(j) for j in range(len(bands.models))])
        for c in grid:
            registered = certificate_upper_bound(bands, np.array([c]), cover, rho)[1]
            assert _slice_certifies(bands, c, cover, rho, lip, 8, 4000) == registered


def test_adaptive_cover_dominates_two_level_and_is_sound():
    from brink.bracket import (
        _slice_certifies,
        _slice_certifies_adaptive,
    )

    grid = np.linspace(0.0, 1.0, 51)
    dense = np.random.default_rng(5).uniform(size=(4000, 2))
    n_adaptive_only = 0
    for bands, n_axis in ((_gap_bands(), 17), (_dense_bands(), 9), (_dense_bands(), 3)):
        cover, rho = midpoint_cover(n_axis, 2)
        lip = np.array([bands.lipschitz(j) for j in range(len(bands.models))])
        for c in grid:
            two = _slice_certifies(bands, c, cover, rho, lip, 8, 4000)
            ada3 = _slice_certifies_adaptive(bands, c, cover, rho, lip, 3, 10**7)
            assert ada3 or not two  # depth 3 reaches the factor-8 subgrid
            ada8 = _slice_certifies_adaptive(bands, c, cover, rho, lip, 8, 10**7)
            n_adaptive_only += ada8 and not two
            if ada8:
                lcb, _ = bands.bands(dense, c)  # every x provably infeasible
                assert np.all(np.any(lcb > 0.0, axis=0))
    assert n_adaptive_only > 0  # deeper levels certify slices factor 8 cannot


def test_adaptive_cover_respects_cell_budget():
    from brink.bracket import _slice_certifies_adaptive

    bands = _dense_bands()
    cover, rho = midpoint_cover(3, 2)
    lip = np.array([bands.lipschitz(0)])
    assert _slice_certifies_adaptive(bands, 1.0, cover, rho, lip, 8, 10**7)
    assert not _slice_certifies_adaptive(bands, 1.0, cover, rho, lip, 8, len(cover))


def test_smallest_scan_adaptive_switch_is_exact_and_off_by_default():
    import pytest

    from brink.bracket import _slice_certifies_adaptive

    grid = np.linspace(0.0, 1.0, 51)
    n_lower = 0
    for bands, n_axis in ((_gap_bands(), 17), (_dense_bands(), 9), (_dense_bands(), 3)):
        cover, rho = midpoint_cover(n_axis, 2)
        lip = np.array([bands.lipschitz(0)])
        two = smallest_certified_slice(bands, grid, cover, rho)
        assert smallest_certified_slice(bands, grid, cover, rho, max_depth=0) == two
        got = smallest_certified_slice(
            bands, grid, cover, rho, max_depth=8, max_cells=10**7
        )
        hits = [
            c for c in grid
            if _slice_certifies_adaptive(bands, c, cover, rho, lip, 8, 10**7)
        ]  # fmt: skip
        assert got == ((float(min(hits)), True) if hits else (1.0, False))
        assert got[0] <= two[0]
        n_lower += got[0] < two[0]
    assert n_lower > 0  # the switch changes the scan somewhere
    bands, cands = _dense_bands(), np.random.default_rng(3).uniform(size=(64, 2))
    cover, rho = midpoint_cover(3, 2)
    off = compute_bracket(bands, cands, grid, cover, rho, scan=SCAN_SMALLEST)
    on = compute_bracket(
        bands, cands, grid, cover, rho, scan=SCAN_SMALLEST, max_depth=8,
        max_cells=10**7,
    )  # fmt: skip
    assert on.upper <= off.upper and on.lower == off.lower
    with pytest.raises(ValueError, match="adaptive cover"):
        compute_bracket(bands, cands, grid, cover, rho, max_depth=8)


def _wavy_bands(mode, n=60, betas=(0.5, 0.5), lam=1e-6, seed=0):
    """Two-constraint toy with real x-dependence (second-order lemma tests)."""
    rng = np.random.default_rng(seed)
    X, c = rng.uniform(size=(n, 2)), rng.uniform(size=n)
    Y = np.stack(
        [
            c - 0.5 + 0.2 * np.sin(4.0 * X[:, 0]) * np.cos(3.0 * X[:, 1]),
            0.4 - c + 0.3 * X[:, 0] - 0.2 * X[:, 1] ** 2,
        ],
        axis=1,
    )
    return fit_joint_bands(
        X, c, Y, mode=mode, lengthscale=0.5, lam=lam, B=np.array([2.0, 2.0]),
        R=0.0, betas=list(betas), c_max=1.0,
    )  # fmt: skip


def _post_cov(bands, j, X1, X2, c):
    """Posterior covariance k_n((x1, c), (x2, c)) by the closed form."""
    from scipy.linalg import solve_triangular

    from brink.bracket import MODE_AFFINE
    from brink.kernels import rbf

    model = bands.models[j]
    if bands.mode == MODE_AFFINE:
        prior = rbf(X1, X2, model.lengthscale) * (1.0 + c * c)
        K1 = rbf(X1, model.X, model.lengthscale) * (1.0 + c * model.c)
        K2 = rbf(X2, model.X, model.lengthscale) * (1.0 + c * model.c)
    else:
        Z1 = np.column_stack([X1, np.full(len(X1), c)])
        Z2 = np.column_stack([X2, np.full(len(X2), c)])
        prior = rbf(Z1, Z2, model.lengthscale)
        K1 = rbf(Z1, model.X, model.lengthscale)
        K2 = rbf(Z2, model.X, model.lengthscale)
    A1 = solve_triangular(model.L, K1.T, lower=True)
    A2 = solve_triangular(model.L, K2.T, lower=True)
    return prior - A1.T @ A2


def test_grad_bands_match_finite_differences():
    from brink.bracket import MODE_AFFINE

    Z = np.random.default_rng(7).uniform(0.1, 0.9, size=(6, 2))
    eps, E = 1e-3, np.eye(2)
    for mode in (MODE_AFFINE, MODE_RBF):
        bands = _wavy_bands(mode, n=40, lam=1e-4)
        for c in (0.3, 0.9):
            grad, s_grad = bands.grad_bands(Z, c)
            assert grad.shape == (2, 6, 2) and s_grad.shape == (2, 6)
            for j in range(2):
                fd = np.stack(
                    [
                        (bands._mu_sd(j, Z + eps * E[a], c)[0]
                         - bands._mu_sd(j, Z - eps * E[a], c)[0]) / (2 * eps)
                        for a in range(2)
                    ],
                    axis=1,
                )  # fmt: skip
                np.testing.assert_allclose(grad[j], fd, rtol=1e-4, atol=1e-5)
                for p, z in enumerate(Z):
                    S = np.empty((2, 2))
                    for a in range(2):
                        for b in range(2):
                            pts = [z + sa * eps * E[a] for sa in (1, -1)]
                            qts = [z + sb * eps * E[b] for sb in (1, -1)]
                            k = _post_cov(bands, j, np.array(pts), np.array(qts), c)
                            S[a, b] = (k[0, 0] - k[0, 1] - k[1, 0] + k[1, 1]) / (
                                4 * eps**2
                            )
                    top = np.linalg.eigvalsh(0.5 * (S + S.T))[-1]
                    np.testing.assert_allclose(
                        s_grad[j, p] ** 2, top, rtol=1e-3, atol=1e-4
                    )


def test_second_order_bound_holds_on_random_pairs():
    """The lemma: lcb(x) >= lcb(z) - r G(z) - r^2 Q for ||x - z|| <= r."""
    from brink.bracket import MODE_AFFINE

    rng = np.random.default_rng(8)
    for mode in (MODE_AFFINE, MODE_RBF):
        bands = _wavy_bands(mode)
        lip = np.array([bands.lipschitz(j) for j in range(2)])
        Q = np.sqrt(3.0) * lip / (2.0 * 0.5)
        b = np.array(bands.betas)
        Z = rng.uniform(size=(400, 2))
        U = rng.normal(size=(400, 2))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        for r in (0.01, 0.1, 0.3):
            X = Z + r * rng.uniform(size=(400, 1)) * U
            for c in (0.2, 0.7):
                lz, _ = bands.bands(Z, c)
                lx, _ = bands.bands(X, c)
                grad, s_grad = bands.grad_bands(Z, c)
                G = np.linalg.norm(grad, axis=2) + b[:, None] * s_grad
                assert np.all(lx >= lz - r * G - r**2 * Q[:, None] - 1e-9)


def test_second_order_or_test_is_sound_and_dominates():
    from brink.bracket import MODE_AFFINE, _slice_certifies_adaptive

    g = (np.arange(200) + 0.5) / 200
    dense = np.stack([m.ravel() for m in np.meshgrid(g, g, indexing="ij")], axis=1)
    grid = np.linspace(0.0, 1.0, 41)
    n_more = 0
    for bands, n_axis, depth in (
        (_wavy_bands(MODE_AFFINE), 5, 3),
        (_wavy_bands(MODE_RBF), 5, 3),
        (_wavy_bands(MODE_AFFINE, betas=(0.05, 0.05)), 3, 4),
        (_gap_bands(), 17, 2),
    ):
        cover, rho = midpoint_cover(n_axis, 2)
        lip = np.array([bands.lipschitz(j) for j in range(len(bands.models))])
        for c in grid:
            old = _slice_certifies_adaptive(bands, c, cover, rho, lip, depth, 10**6)
            new = _slice_certifies_adaptive(
                bands, c, cover, rho, lip, depth, 10**6, second_order=True
            )
            assert new or not old  # OR test certifies everything the old one does
            n_more += new and not old
            if new:  # never certifies a slice with a dense point not cleared
                lcb, _ = bands.bands(dense, c)
                assert np.max(lcb, axis=0).min() > 0.0
    assert n_more > 0  # and it certifies slices the first-order test cannot


def test_second_order_switch_is_off_by_default_and_bit_identical():
    """Pinned brackets from before the switch existed."""
    import pytest

    grid = np.linspace(0.0, 1.0, 51)
    cands = np.random.default_rng(3).uniform(size=(64, 2))
    adaptive = dict(scan=SCAN_SMALLEST, max_depth=8, max_cells=10**7)
    pinned = {
        (17, 0): (0.58, 1.0, True, False),
        (17, 1): (0.58, 0.68, True, True),
        (17, 2): (0.58, 0.62, True, True),
        (9, 0): (0.58, 0.74, True, True),
        (9, 1): (0.58, 0.74, True, True),
        (9, 2): (0.58, 0.64, True, True),
        (3, 0): (0.58, 1.0, True, False),
        (3, 1): (0.58, 1.0, True, False),
        (3, 2): (0.58, 0.66, True, True),
    }
    for bands, n in ((_gap_bands(), 17), (_dense_bands(), 9), (_dense_bands(), 3)):
        cover, rho = midpoint_cover(n, 2)
        for k, kw in enumerate(({}, dict(scan=SCAN_SMALLEST), adaptive)):
            br = compute_bracket(bands, cands, grid, cover, rho, **kw)
            got = (br.lower, br.upper, br.has_witness, br.has_certificate)
            assert got == pinned[(n, k)]
    with pytest.raises(ValueError, match="second-order"):
        compute_bracket(bands, cands, grid, cover, rho, scan=SCAN_SMALLEST,
                        second_order=True)  # fmt: skip
    on = compute_bracket(bands, cands, grid, cover, rho, second_order=True, **adaptive)
    assert on.upper <= pinned[(3, 2)][1]


def test_second_order_clearance_flips_exactly_at_the_lemma_radius():
    """Pins the production margin lcb - r G - r^2 Q (review T1, 2026-10).

    One cell, no refinement, first-order margin negative: the slice certifies
    iff r < r*, the largest root of lcb_j(z) - r G_j - r^2 Q_j = 0 over j.
    """
    from brink.bracket import MODE_AFFINE, _slice_certifies_adaptive

    cases = ((MODE_AFFINE, (0.133, 0.113), 0.2), (MODE_RBF, (0.61, 0.316), 0.2))
    for mode, z, c in cases:
        bands = _wavy_bands(mode)
        lip = np.array([bands.lipschitz(j) for j in range(2)])
        Q = np.sqrt(3.0) * lip / (2.0 * 0.5)
        Z = np.array([z])
        lcb = bands.bands(Z, c)[0][:, 0]
        grad, s_grad = bands.grad_bands(Z, c)
        G = np.linalg.norm(grad[:, 0], axis=1) + np.array(bands.betas) * s_grad[:, 0]
        pos = np.maximum(lcb, 0.0)
        r_star = np.max((-G + np.sqrt(G**2 + 4.0 * Q * pos)) / (2.0 * Q))
        assert r_star > 1.5 * np.max(pos / lip)  # first-order test cannot clear
        assert np.max(bands.betas * s_grad[:, 0] / G) > 0.05  # b s_grad matters
        for r, want in ((r_star * (1 - 1e-6), True), (r_star * (1 + 1e-6), False)):
            got = _slice_certifies_adaptive(
                bands, c, Z, r, lip, 0, 10, second_order=True
            )
            assert got is want


def test_second_order_does_not_clear_a_concave_bump():
    """lcb peaks at the cell centre (grad ~ 0) and is negative inside the cell."""
    from brink.bracket import _slice_certifies_adaptive

    X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    y = np.array([-1.0, -0.2, 0.05, -0.2, -1.0])
    bands = fit_joint_bands(
        X, np.full(5, 0.5), y[:, None], mode=MODE_RBF, lengthscale=0.15, lam=1e-6,
        B=np.ones(1), R=0.0, betas=[0.0], c_max=1.0,
    )  # fmt: skip
    lip = np.array([bands.lipschitz(0)])
    lcb, _ = bands.bands(np.linspace(0.3, 0.7, 401)[:, None], 0.5)
    assert lcb.min() < 0.0 < bands.bands(np.array([[0.5]]), 0.5)[0][0, 0]
    for depth in (0, 6):
        assert not _slice_certifies_adaptive(
            bands, 0.5, np.array([[0.5]]), 0.2, lip, depth, 10**5, second_order=True
        )


def test_certificate_tests_reject_affine_slices_beyond_c_max():
    """kappa = sqrt(1 + c_max^2) bounds kappa_c only for |c| <= c_max (flag 3)."""
    import pytest

    from brink.bracket import (
        MODE_AFFINE,
        _slice_certifies,
        _slice_certifies_adaptive,
    )

    cover, rho = midpoint_cover(3, 2)
    for mode in (MODE_AFFINE, MODE_RBF):
        bands = _wavy_bands(mode)
        lip = np.array([bands.lipschitz(j) for j in range(2)])
        calls = (
            lambda c, b=bands: certificate_upper_bound(
                b, np.array([0.5, c]), cover, rho
            ),
            lambda c, b=bands: smallest_certified_slice(
                b, np.array([0.5, c]), cover, rho
            ),
            lambda c, b=bands, L=lip: _slice_certifies(b, c, cover, rho, L, 8, 4000),
            lambda c, b=bands, L=lip: _slice_certifies_adaptive(
                b, c, cover, rho, L, 2, 9
            ),
        )
        for call in calls:
            call(1.0)
            call(-1.0)
            if mode == MODE_AFFINE:
                with pytest.raises(ValueError, match="c_max"):
                    call(1.01)
            else:
                call(1.01)  # kappa_c = 1 in the concatenated-RBF mode


def test_smallest_scan_time_cap_and_scan_off():
    """A zero time cap returns the trivial endpoint; scan 'off' runs no certificate."""
    from brink.bracket import SCAN_OFF

    bands, grid = _gap_bands(), np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(17, 2)
    assert smallest_certified_slice(bands, grid, cover, rho)[1]
    assert smallest_certified_slice(bands, grid, cover, rho, time_cap_s=0.0) == (
        1.0,
        False,
    )
    cands = np.random.default_rng(3).uniform(size=(64, 2))
    off = compute_bracket(bands, cands, grid, cover, rho, scan=SCAN_OFF, max_depth=12)
    on = compute_bracket(bands, cands, grid, cover, rho, scan=SCAN_SMALLEST)
    assert not off.has_certificate and off.upper == 1.0 and on.has_certificate
    assert (off.lower, off.has_witness) == (on.lower, on.has_witness)


def test_top_down_scan_adaptive_form_and_time_cap():
    """S3-A3: the top-down scan takes the adaptive cell test and a time cap."""
    bands, grid = _dense_bands(), np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(9, 2)
    two_level = certificate_upper_bound(bands, grid, cover, rho)
    adaptive = certificate_upper_bound(bands, grid, cover, rho, max_depth=8)
    assert two_level[1] and adaptive[1]
    assert adaptive[0] <= two_level[0]  # the adaptive test is never worse
    # Under nesting both adaptive scans find the same top interval here.
    assert adaptive == smallest_certified_slice(bands, grid, cover, rho, max_depth=8)
    gap = _gap_bands()
    cover_g, rho_g = midpoint_cover(17, 2)
    down = certificate_upper_bound(gap, grid, cover_g, rho_g, max_depth=8)
    assert down == smallest_certified_slice(gap, grid, cover_g, rho_g, max_depth=8)
    assert down[1] and down[0] >= 0.6  # sound: theta = 0.6 on the gap bands
    # The cap returns what has been certified so far (a valid, non-smallest bound).
    assert certificate_upper_bound(bands, grid, cover, rho, time_cap_s=0.0) == (
        1.0,
        False,
    )
    import pytest

    with pytest.raises(ValueError, match="second-order"):
        certificate_upper_bound(bands, grid, cover, rho, second_order=True)


def _window_bands():
    """Two-sided toy: every x is feasible exactly on [0.40, 0.80].

    Channels g_lo = 0.40 - c and g_hi = c - 0.80 (the 2026-11-08 proof-audit
    counterexample): slices below 0.40 are empty, yet theta_exists = 0.80.
    """
    rng = np.random.default_rng(0)
    X, c = rng.uniform(size=(300, 2)), rng.uniform(size=300)
    return fit_joint_bands(
        X,
        c,
        np.column_stack([0.40 - c, c - 0.80]),
        mode=MODE_RBF,
        lengthscale=0.5,
        lam=1e-6,
        B=np.array([2.0, 2.0]),
        R=0.0,
        betas=[1.0, 1.0],
        c_max=1.0,
    )


def test_upper_channel_scan_is_sound_on_two_sided_windows():
    bands, grid = _window_bands(), np.linspace(0.0, 1.0, 51)
    cover, rho = midpoint_cover(9, 2)
    both, ok = smallest_certified_slice(bands, grid, cover, rho)
    assert ok and both < 0.40  # an empty slice below the window bounds nothing
    upper_only = bands.subset([1])
    assert len(upper_only.models) == 1 and upper_only.betas == [1.0]
    for kw in ({}, {"max_depth": 8, "max_cells": 10**7}):
        upper, ok = smallest_certified_slice(upper_only, grid, cover, rho, **kw)
        assert ok and upper > 0.80  # sound: g_hi > 0 persists upward


def test_offsets_shift_bands_by_exactly_eta():
    bands = _dense_bands(80)
    X = np.random.default_rng(7).uniform(size=(50, 2))
    lcb0, ucb0 = bands.bands(X, 0.4)
    bands.offsets = [0.0]
    lcb_z, ucb_z = bands.bands(X, 0.4)
    assert np.array_equal(lcb0, lcb_z) and np.array_equal(ucb0, ucb_z)
    bands.offsets = [0.1]
    lcb1, ucb1 = bands.bands(X, 0.4)
    assert np.allclose(lcb0 - lcb1, 0.1, atol=1e-14)
    assert np.allclose(ucb1 - ucb0, 0.1, atol=1e-14)
    assert bands.subset([0]).offsets == [0.1]
    c_arr = np.full(50, 0.62)
    bands.offsets = None
    keep0 = bands.undetermined(X, c_arr)
    bands.offsets = [0.1]
    assert np.all(bands.undetermined(X, c_arr) >= keep0)


def test_offsets_never_narrow_the_bracket():
    c_grid = np.linspace(0.0, 1.0, 101)
    cands = np.random.default_rng(8).uniform(size=(256, 2))
    cover, rho = midpoint_cover(41, 2)
    bands = _dense_bands(200)
    br0 = compute_bracket(bands, cands, c_grid, cover, rho)
    bands.offsets = [0.05]
    br1 = compute_bracket(bands, cands, c_grid, cover, rho)
    assert br1.lower <= br0.lower and br1.upper >= br0.upper
    assert br1.lower <= 0.6 <= br1.upper
