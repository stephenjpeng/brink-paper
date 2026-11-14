"""Transmission-network study: IEEE 39-bus loadability (pandapower ``case39``).

Design x (4 dims, unit box): active-power setpoints of three generators
(pandapower gen rows 0, 5, 6 at case39 buses 30, 36, 37) scaled in [0.5, 1.5]
of nominal, plus an inductive shunt reactor of 0 to 300 MVAr at pandapower
bus 24. Stress c: a uniform load multiplier in [1.0, 1.35] applied to every
active and reactive load.

A point is feasible when the Newton power flow converges, every PQ
(non-generator) bus voltage lies in [0.94, 1.06] pu, and no line or
transformer loading exceeds 100%. Solver: Newton-Raphson from a DC start,
``enforce_q_lims=True``, numba off, so calls are history-free and
deterministic.

The model sees the window representation (paper Section 5, Appendix C.2):

    g_lo = c_on(x) - c    and    g_hi = c - c_off(x),

in unit c. One window call evaluates the raw predicate at a 9-point coarse
scan plus 8 bisection steps per window edge (at most 32 power flows).
Standardization is scale-only and unclipped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brink.window_margins import WindowMarginOracle

logger = logging.getLogger(__name__)

# Environment constants.
GEN_ROWS = (0, 5, 6)
GEN_BUSES_CASE39 = (30, 36, 37)  # 1-based case39 bus numbers, for the record
GEN_SCALE_RANGE = (0.5, 1.5)
SHUNT_BUS = 24
SHUNT_RANGE_MVAR = (0.0, 300.0)
C_RANGE = (1.0, 1.35)
V_BAND = (0.94, 1.06)
CRASH_MARGIN = 3.0
M_CHANNELS = 3  # raw margins of one power flow: low voltage, high voltage, loading
X_DIM = 4
SOLVER_SETTINGS = {
    "algorithm": "nr",
    "init": "dc",
    "enforce_q_lims": True,
    "numba": False,
}


class GridLoadabilityOracle:
    """Raw-margin oracle: one Newton power flow per call.

    Holds one mutable case39 network per instance; every varying quantity
    (loads, generator setpoints, shunt MVAr) is rewritten on
    each call and the solver starts from a DC initialization, so calls are
    history-free and deterministic.
    """

    def __init__(self) -> None:
        """Build the case39 network with the shunt element."""
        import pandapower as pp
        import pandapower.networks as pn

        self._pp = pp
        self.net = pn.case39()
        self.shunt_idx = pp.create_shunt(self.net, bus=SHUNT_BUS, q_mvar=0.0)
        self.base_p = self.net.load.p_mw.to_numpy().copy()
        self.base_q = self.net.load.q_mvar.to_numpy().copy()
        self.gen_nominal = self.net.gen.p_mw.to_numpy().copy()
        pv_buses = sorted(
            set(self.net.gen.bus.tolist()) | set(self.net.ext_grid.bus.tolist())
        )
        self.pq_mask = ~self.net.bus.index.isin(pv_buses)

    @staticmethod
    def to_physical(x_unit: np.ndarray, c_unit: float) -> tuple[np.ndarray, float]:
        """Map unit design coordinates to physical values.

        Args:
            x_unit: Unit design point, shape (4,): three generator scales
                and the shunt coordinate.
            c_unit: Unit coefficient.

        Returns:
            Tuple (x_phys, c_phys): generator scales in [0.5, 1.5], shunt
            MVAr in [0, 300], load factor in [1.0, 1.35].
        """
        x_unit = np.asarray(x_unit, dtype=float)
        s_lo, s_hi = GEN_SCALE_RANGE
        q_lo, q_hi = SHUNT_RANGE_MVAR
        c_lo, c_hi = C_RANGE
        x_phys = np.empty(X_DIM)
        x_phys[:3] = s_lo + (s_hi - s_lo) * x_unit[:3]
        x_phys[3] = q_lo + (q_hi - q_lo) * x_unit[3]
        return x_phys, c_lo + (c_hi - c_lo) * float(c_unit)

    def raw_margins(self, x_unit: np.ndarray, c_unit: float) -> np.ndarray | None:
        """Raw margins at a unit-coordinate query point.

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.

        Returns:
            Raw margins (g1, g2, g3), or None when the power flow does not
            converge (crash convention applies upstream).
        """
        x_phys, c_phys = self.to_physical(x_unit, c_unit)
        net = self.net
        net.load["p_mw"] = self.base_p * c_phys
        net.load["q_mvar"] = self.base_q * c_phys
        for gi, scale in zip(GEN_ROWS, x_phys[:3], strict=True):
            net.gen.at[gi, "p_mw"] = self.gen_nominal[gi] * scale
        net.shunt.at[self.shunt_idx, "q_mvar"] = x_phys[3]
        try:
            self._pp.runpp(net, **SOLVER_SETTINGS)
        except Exception:
            return None
        v_pq = net.res_bus.vm_pu.to_numpy()[self.pq_mask]
        loading = max(
            float(net.res_line.loading_percent.max()),
            float(net.res_trafo.loading_percent.max()),
        )
        return np.array(
            [
                V_BAND[0] - float(v_pq.min()),
                float(v_pq.max()) - V_BAND[1],
                (loading - 100.0) / 100.0,
            ]
        )

    def feasible(self, x_unit: np.ndarray, c_unit: float) -> bool:
        """Whether the point is feasible (converged and all margins <= 0).

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.

        Returns:
            True when the power flow converges and every margin is <= 0.
        """
        g = self.raw_margins(x_unit, c_unit)
        return g is not None and bool((g <= 0.0).all())


M_CHANNELS_WINDOW = 2


def make_window_oracle(
    raw: GridLoadabilityOracle | None = None,
) -> WindowMarginOracle:
    """Window-margin oracle over the v1 raw feasibility predicate.

    Args:
        raw: Existing raw power-flow oracle to wrap (built when None).

    Returns:
        WindowMarginOracle with m = 2 (the transmission study has
        delayed-onset columns, so the family is not a pure prefix).
    """
    impl = raw if raw is not None else GridLoadabilityOracle()
    return WindowMarginOracle(impl.feasible, prefix_only=False)


@dataclass
class TransmissionEnvironment:
    """Frozen transmission-network environment (window representation, m = 2).

    Attributes:
        scaler_scale: Per-channel standardization scale IQR / 1.349, (2,).
        B: Frozen inflated joint-RKHS norm bounds, shape (2,).
        B_slice: Frozen inflated per-slice norm bounds, shape (2,).
        window_oracle: Lazily built window oracle (one network/process).
    """

    scaler_scale: np.ndarray
    B: np.ndarray
    B_slice: np.ndarray
    window_oracle: WindowMarginOracle | None = field(default=None, repr=False)

    x_dim = X_DIM
    m = M_CHANNELS_WINDOW
    noise_knob = None
    x_bounds = (
        GEN_SCALE_RANGE,
        GEN_SCALE_RANGE,
        GEN_SCALE_RANGE,
        SHUNT_RANGE_MVAR,
    )
    c_range = C_RANGE

    def _oracle(self) -> WindowMarginOracle:
        """The per-instance window oracle, built on first use."""
        if self.window_oracle is None:
            self.window_oracle = make_window_oracle()
        return self.window_oracle

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Standardized window margins at unit (x, c); deterministic.

        Args:
            x: Unit design point, shape (4,).
            c: Unit coefficient.
            rng: Unused (deterministic environment).
            sigma: Must be 0.0; this environment has no noise knob.

        Returns:
            Standardized channels (g_lo, g_hi) / scale, shape (2,),
            unclipped (exactly affine in c).

        Raises:
            ValueError: When sigma is nonzero.
        """
        if sigma != 0.0:
            raise ValueError("the transmission study is deterministic; sigma must be 0")
        raw = self._oracle().raw_channels(np.asarray(x, dtype=float), float(c))
        return raw / self.scaler_scale

    def window_counters(self) -> dict:
        """Raw-evaluation accounting of the window oracle."""
        if self.window_oracle is None:
            return {
                "n_calls": 0,
                "n_raw_evals": 0,
                "raw_evals_per_call": 0.0,
                "n_multi_window": 0,
            }
        return self.window_oracle.counters()


def load_environment(path: str | Path) -> TransmissionEnvironment:
    """Load the frozen transmission environment from its JSON file.

    Args:
        path: Path to ``configs/frozen/transmission.json``.

    Returns:
        TransmissionEnvironment with frozen scalers and norm bounds.

    Raises:
        ValueError: When the frozen registration disagrees with the module
            constants (guards against silent drift).
    """
    payload = json.loads(Path(path).read_text())
    env_reg = payload["environment"]
    expected = {
        "gen_rows": list(GEN_ROWS),
        "shunt_bus": SHUNT_BUS,
        "c_range": list(C_RANGE),
        "oracle_representation": "window-margin",
        "m": M_CHANNELS_WINDOW,
    }
    for key, value in expected.items():
        if env_reg[key] != value:
            raise ValueError(
                f"frozen registration mismatch for {key}: "
                f"file has {env_reg[key]!r}, module has {value!r}"
            )
    cal = payload["calibration"]
    return TransmissionEnvironment(
        scaler_scale=np.asarray(cal["scaler_scale"], dtype=float),
        B=np.asarray(cal["B"], dtype=float),
        B_slice=np.asarray(cal["B_slice"], dtype=float),
    )
