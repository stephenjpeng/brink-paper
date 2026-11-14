"""Tests for the double-integrator viability environment."""

from __future__ import annotations

import numpy as np
import pytest

from brink.studies.double_integrator import (
    C_RANGE,
    P_MAX,
    THETA_EXISTS_UNIT,
    U_MAX,
    V_MAX,
    DoubleIntegratorEnvironment,
    c_off_physical,
    c_off_unit,
    is_viable,
    theta_exists_truth,
)


def _simulate_worst_case(p0: float, v0: float, c: float, dt: float = 2e-5) -> bool:
    """Independent reference: simulate the worst case and report viability.

    The adversary pushes with ``w = +c`` and the controller opposes with
    ``u = -u_max``, so the net acceleration is ``c - u_max``. The state is
    viable when the position stays within the box until the velocity reaches
    zero. This knows nothing about the closed form under test.
    """
    if c >= U_MAX:
        # Net acceleration is non-negative: the velocity never decreases, so
        # the position grows without bound unless it is already at rest at
        # exactly the neutral point.
        return bool(v0 <= 0.0 and c == U_MAX and p0 <= 1.0)
    p, v = p0, v0
    accel = c - U_MAX
    for _ in range(int(5e6)):
        if p > 1.0:
            return False
        if v <= 0.0:
            return True
        v += accel * dt
        p += v * dt
    return p <= 1.0


@pytest.mark.parametrize(
    ("p0", "v0"),
    [(0.0, 0.2), (0.25, 0.5), (0.4, 0.8), (0.5, 1.0), (0.1, 0.05), (0.3, 0.0)],
)
def test_closed_form_c_off_matches_worst_case_simulation(p0: float, v0: float) -> None:
    """The closed form agrees with an independent worst-case simulation."""
    x_unit = np.array([[p0 / P_MAX, v0 / V_MAX]])
    c_star = float(c_off_physical(x_unit)[0])
    # Just inside the closed-form boundary the state must survive.
    if c_star - 1e-3 > 0.0:
        assert _simulate_worst_case(p0, v0, c_star - 1e-3)
    # Just outside it must not, unless the column is viable at every level.
    if c_star + 1e-3 < U_MAX:
        assert not _simulate_worst_case(p0, v0, c_star + 1e-3)


def test_theta_exists_is_exact_and_attained_on_the_face() -> None:
    """Truth is exact, and the midpoint grid alone underestimates it."""
    truth = theta_exists_truth(n_per_axis=401)
    assert truth["stated_error"] == 0.0
    assert truth["theta_unit_analytic"] == pytest.approx(1.0 / 1.5, abs=1e-12)
    # The face carries the supremum; a midpoint grid never lands on it.
    assert truth["theta_unit_including_face"] == pytest.approx(
        truth["theta_unit_analytic"], abs=1e-12
    )
    assert truth["grid_underestimate"] > 0.0
    assert truth["grid_underestimate"] < 1e-5


def test_zero_columns_are_viable_at_c_max() -> None:
    """The crossing lies inside the range (the check S6 failed)."""
    truth = theta_exists_truth(n_per_axis=401)
    assert truth["columns_viable_at_c_max"] == 0
    assert THETA_EXISTS_UNIT < 1.0


def test_family_is_a_prefix_so_m_is_one() -> None:
    """Every column is viable at the smallest coefficient."""
    truth = theta_exists_truth(n_per_axis=401)
    assert truth["columns_viable_at_c_min"] == truth["n_columns"]


def test_family_is_nested_in_c() -> None:
    """Viability is monotone decreasing in the coefficient."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(400, 2))
    grid = np.linspace(0.0, 1.0, 41)
    viable = np.stack([is_viable(X, c) for c in grid])
    # Once a column dies it never revives.
    assert np.all(viable[:-1] >= viable[1:])


def test_standardized_channel_is_exactly_affine_in_c() -> None:
    """The channel is affine in c, which the joint kernel requires."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    rng = np.random.default_rng(1)
    for x in rng.uniform(size=(20, 2)):
        g0 = float(env.oracle(x, 0.0)[0])
        g1 = float(env.oracle(x, 1.0)[0])
        for c in (0.13, 0.5, 0.77):
            expected = g0 + c * (g1 - g0)
            assert float(env.oracle(x, c)[0]) == pytest.approx(expected, abs=1e-12)


def test_channel_sign_matches_the_viability_predicate() -> None:
    """A non-positive channel means viable, which the arms rely on."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    rng = np.random.default_rng(2)
    for x in rng.uniform(size=(50, 2)):
        for c in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert bool(env.oracle(x, c)[0] <= 0.0) == bool(is_viable(x, c)[0])


def test_oracle_counts_one_raw_evaluation_per_call() -> None:
    """Raw-evaluation accounting is 1:1 for this closed-form oracle."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    assert env.window_counters()["n_raw_evals"] == 0
    for _ in range(7):
        env.oracle(np.array([0.3, 0.4]), 0.5)
    assert env.window_counters()["n_raw_evals"] == 7


def test_oracle_rejects_noise() -> None:
    """The environment is deterministic and says so."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    with pytest.raises(ValueError, match="deterministic"):
        env.oracle(np.array([0.3, 0.4]), 0.5, np.random.default_rng(0), 0.01)


def test_true_volume_decreases_and_vanishes() -> None:
    """The volume curve collapses to zero inside the range."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    vols = [env.true_volume(c, n_per_axis=201) for c in np.linspace(0.0, 1.0, 21)]
    assert vols[0] == pytest.approx(1.0)
    assert vols[-1] == 0.0
    assert all(a >= b - 1e-12 for a, b in zip(vols[:-1], vols[1:], strict=True))


def test_score_matches_c_off() -> None:
    """The reduced score is c_off, which is what makes the oracle reducible."""
    env = DoubleIntegratorEnvironment(
        scaler_scale=np.array([0.2]),
        B=np.array([50.0]),
        B_slice=np.array([25.0]),
        lengthscale=1.0,
    )
    rng = np.random.default_rng(3)
    for x in rng.uniform(size=(20, 2)):
        assert env.score_unit(x) == pytest.approx(float(c_off_unit(x)[0]), abs=1e-12)


def test_registered_box_is_away_from_the_pole() -> None:
    """The restriction is part of the environment definition."""
    assert P_MAX == 0.5
    assert V_MAX == 1.0
    assert C_RANGE == (0.0, 1.5)
