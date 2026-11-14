"""Tests for the transmission-network raw power-flow oracle."""

import numpy as np
import pytest

from brink.studies.transmission import C_RANGE, GridLoadabilityOracle


def test_to_physical_mapping():
    x_phys, c_phys = GridLoadabilityOracle.to_physical(
        np.array([0.0, 0.5, 1.0, 0.5]), 0.0
    )
    np.testing.assert_allclose(x_phys, [0.5, 1.0, 1.5, 150.0])
    assert c_phys == pytest.approx(C_RANGE[0])
    _, c_hi = GridLoadabilityOracle.to_physical(np.zeros(4), 1.0)
    assert c_hi == pytest.approx(C_RANGE[1])


def test_real_oracle_deterministic_and_history_free():
    oracle = GridLoadabilityOracle()
    x_a = np.array([0.5, 0.5, 0.5, 0.0])
    x_b = np.array([0.9, 0.1, 0.7, 0.8])
    first = oracle.raw_margins(x_a, 0.1)
    _ = oracle.raw_margins(x_b, 0.9)  # interleaved different state
    second = oracle.raw_margins(x_a, 0.1)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3,)


def test_real_oracle_margins_move_with_load():
    oracle = GridLoadabilityOracle()
    x = np.full(4, 0.5)
    g_low = oracle.raw_margins(x, 0.0)
    g_high = oracle.raw_margins(x, 0.3)
    # Undervoltage margin g1 worsens (rises) with load growth.
    assert g_high[0] > g_low[0]
