"""Distribution-network study: solar hosting on the CIGRE MV feeder (Appendix C.5).

Network: ``pandapower.networks.create_cigre_network_mv(with_der="pv_wind")``.

- Design x (4 dims, unit box): solar plant sizes in [0, 4] MW at the 4 load
  buses with the highest total nominal load (buses 12, 1, 5, 8).
- Stress c: load-growth factor in [1.0, 2.0] scaling all loads.
- Operating conditions (seeded by the call RNG): each condition draws one
  lognormal multiplier per load with sigma_ln = 0.35 and mean one, applied to
  P and Q, and an output uniform in [0.2, 1.0] of its size for each plant;
  the network's own distributed generators stay at nominal output.
- Raw margins per condition (m = 2): g1 = max over buses of |V - 1.0| - 0.06,
  g2 = max over lines and transformers of (loading - 100) / 100. A
  non-converged condition contributes +3.0 to both.
- One check: the means of the margins over K conditions are both <= 0; the
  reported study uses K = 16.

The model sees window observations over this check (``window_margins``),
with fresh conditions for every check. Power-flow fast path: the topology,
admittance matrices, and branch ratings are fixed, so each condition re-solves
a pure-PQ Newton power flow on the pandapower-built Ybus/Yf/Yt (flat start,
tolerance 1e-8 MVA, at most 10 iterations, matching ``pp.runpp`` with the
same settings); ``verify_fast_path`` checks the agreement.
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
NETWORK_NAME = "pandapower.networks.create_cigre_network_mv(with_der='pv_wind')"
WITH_DER = "pv_wind"
N_PV_PLANTS = 4
PV_SIZE_RANGE_MW = (0.0, 4.0)
C_RANGE = (1.0, 2.0)
SIGMA_LN = 0.35
PV_OUTPUT_RANGE = (0.2, 1.0)
V_DEV_LIMIT = 0.06
SNAPSHOT_CRASH_MARGIN = 3.0
M_CHANNELS = 2
X_DIM = 4
NOISE_KNOBS = (16, 4, 1)
# The model uses the upper-window channel alone (m = 1): 4 of the 2,401 grid
# designs of the reference search have a delayed onset (Appendix C.2).
REPORT_GRID_SIZE = 201
PREFIX_NO_WINDOW_REPORT_GRID_STEP = 1.0 / (REPORT_GRID_SIZE - 1)
# Solver settings (fast path mirrors pp.runpp with these).
SOLVER_SETTINGS = {
    "algorithm": "nr",
    "init": "flat",
    "enforce_q_lims": False,
    "numba": False,
    "tolerance_mva": 1.0e-8,
}
NEWTON_TOL = 1.0e-8
NEWTON_MAX_IT = 10


def select_pv_buses(net) -> list[int]:
    """The 4 load buses with the highest total nominal load (descending).

    Args:
        net: pandapower network.

    Returns:
        List of 4 pandapower bus indices, highest total nominal p_mw first.
    """
    agg = net.load.groupby("bus")["p_mw"].sum().sort_values(ascending=False)
    return [int(b) for b in agg.index[:N_PV_PLANTS]]


class HostingCapacityOracle:
    """Raw snapshot oracle with the validated fixed-Ybus Newton fast path.

    Holds one CIGRE MV network per instance. The pandapower network is
    built once (with the 4 PV sgens added), ``pp.runpp`` runs
    once to populate the internal ppc, and every snapshot afterwards
    re-solves the pure-PQ Newton power flow directly on the fixed
    admittance matrices.

    Attributes:
        pv_buses: PV plant buses (4 highest-load buses).
    """

    def __init__(self, pv_buses: list[int] | None = None) -> None:
        """Build the network, add the PV sgens, and freeze the fast path.

        Args:
            pv_buses: PV bus list; computed from the network
                when None.

        Raises:
            ValueError: When a provided bus list disagrees with the
                highest-load selection rule.
        """
        import pandapower as pp
        import pandapower.networks as pn

        self._pp = pp
        net = pn.create_cigre_network_mv(with_der=WITH_DER)
        self.net = net
        selected = select_pv_buses(net)
        if pv_buses is not None and list(pv_buses) != selected:
            raise ValueError(
                f"registered PV buses {pv_buses} disagree with the "
                f"highest-load selection {selected}"
            )
        self.pv_buses = selected
        self.pv_idx = [
            pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0) for b in self.pv_buses
        ]
        pp.runpp(net, **SOLVER_SETTINGS)
        ppc = net._ppc
        self.Ybus = ppc["internal"]["Ybus"].toarray()
        self.Yf = ppc["internal"]["Yf"].toarray()
        self.Yt = ppc["internal"]["Yt"].toarray()
        self.n_ppc = self.Ybus.shape[0]
        self.base_mva = float(ppc["baseMVA"])
        bus_lookup = np.asarray(net._pd2ppc_lookups["bus"])
        self.slack_ppc = int(bus_lookup[net.ext_grid.bus.iloc[0]])
        self.slack_vm = float(net.ext_grid.vm_pu.iloc[0])
        self.nonslack = np.array([i for i in range(self.n_ppc) if i != self.slack_ppc])
        self.load_bus_ppc = bus_lookup[net.load.bus.to_numpy()]
        self.base_p = net.load.p_mw.to_numpy().copy()
        self.base_q = net.load.q_mvar.to_numpy().copy()
        fixed_sgen = [i for i in net.sgen.index if i not in self.pv_idx]
        self.fixed_sgen_bus_ppc = bus_lookup[net.sgen.bus.loc[fixed_sgen].to_numpy()]
        self.fixed_sgen_p = net.sgen.p_mw.loc[fixed_sgen].to_numpy().copy()
        self.pv_bus_ppc = bus_lookup[np.array(self.pv_buses)]
        fb = ppc["branch"][:, 0].real.astype(int)
        tb = ppc["branch"][:, 1].real.astype(int)
        vn_ppc = ppc["bus"][:, 9].real  # BASE_KV per ppc bus
        self.vn_f = vn_ppc[fb]
        self.vn_t = vn_ppc[tb]
        l0, l1 = net._pd2ppc_lookups["branch"]["line"]
        t0, t1 = net._pd2ppc_lookups["branch"]["trafo"]
        self.line_rows = np.arange(l0, l1)
        self.trafo_rows = np.arange(t0, t1)
        self.line_i_rated = (
            net.line.max_i_ka * net.line.df * net.line.parallel
        ).to_numpy()
        self.trafo_i_rated_hv = (
            net.trafo.sn_mva / (np.sqrt(3.0) * net.trafo.vn_hv_kv)
        ).to_numpy()
        self.trafo_i_rated_lv = (
            net.trafo.sn_mva / (np.sqrt(3.0) * net.trafo.vn_lv_kv)
        ).to_numpy()
        self.pd_bus_ppc = bus_lookup[net.bus.index.to_numpy()]

    @staticmethod
    def to_physical(x_unit: np.ndarray, c_unit: float) -> tuple[np.ndarray, float]:
        """Map unit design coordinates to physical values.

        Args:
            x_unit: Unit design point, shape (4,): PV sizes.
            c_unit: Unit coefficient.

        Returns:
            Tuple (pv_sizes_mw in [0, 4], load factor in [1.0, 2.0]).
        """
        s_lo, s_hi = PV_SIZE_RANGE_MW
        c_lo, c_hi = C_RANGE
        sizes = s_lo + (s_hi - s_lo) * np.asarray(x_unit, dtype=float)
        return sizes, c_lo + (c_hi - c_lo) * float(c_unit)

    def _solve_fast(
        self, load_p: np.ndarray, load_q: np.ndarray, pv_p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Fixed-Ybus pure-PQ Newton power flow (validated fast path).

        Args:
            load_p: Per-load active power, MW.
            load_q: Per-load reactive power, MVAr.
            pv_p: Per-plant PV output, MW (bus order).

        Returns:
            Tuple (vm_pu per bus, line loading %, trafo loading %), or
            None on non-convergence.
        """
        S = np.zeros(self.n_ppc, dtype=complex)
        np.add.at(S, self.load_bus_ppc, -(load_p + 1j * load_q))
        np.add.at(S, self.fixed_sgen_bus_ppc, self.fixed_sgen_p)
        np.add.at(S, self.pv_bus_ppc, pv_p)
        Sbus = S / self.base_mva
        ns = self.nonslack
        Y = self.Ybus
        V = np.full(self.n_ppc, 1.0 + 0j)
        V[self.slack_ppc] = self.slack_vm
        for _ in range(NEWTON_MAX_IT):
            Ibus = Y @ V
            mis = V * np.conj(Ibus) - Sbus
            F = np.concatenate([mis[ns].real, mis[ns].imag])
            if np.max(np.abs(F)) < NEWTON_TOL:
                break
            diagV = np.diag(V)
            diagI = np.diag(Ibus)
            diagVnorm = np.diag(V / np.abs(V))
            dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
            dS_dVm = diagV @ np.conj(Y @ diagVnorm) + np.conj(diagI) @ diagVnorm
            J = np.block(
                [
                    [dS_dVa[np.ix_(ns, ns)].real, dS_dVm[np.ix_(ns, ns)].real],
                    [dS_dVa[np.ix_(ns, ns)].imag, dS_dVm[np.ix_(ns, ns)].imag],
                ]
            )
            try:
                dx = np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                return None
            k = len(ns)
            Va = np.angle(V)
            Vm = np.abs(V)
            Va[ns] -= dx[:k]
            Vm[ns] -= dx[k:]
            V = Vm * np.exp(1j * Va)
        mis = V * np.conj(Y @ V) - Sbus
        if np.max(np.abs(np.concatenate([mis[ns].real, mis[ns].imag]))) >= NEWTON_TOL:
            return None
        vm = np.abs(V)[self.pd_bus_ppc]
        i_f = np.abs(self.Yf @ V) * self.base_mva / (np.sqrt(3.0) * self.vn_f)
        i_t = np.abs(self.Yt @ V) * self.base_mva / (np.sqrt(3.0) * self.vn_t)
        line_loading = (
            np.maximum(i_f[self.line_rows], i_t[self.line_rows])
            / self.line_i_rated
            * 100.0
        )
        trafo_loading = (
            np.maximum(
                i_f[self.trafo_rows] / self.trafo_i_rated_hv,
                i_t[self.trafo_rows] / self.trafo_i_rated_lv,
            )
            * 100.0
        )
        return vm, line_loading, trafo_loading

    def _snapshot_inputs(
        self, x_unit: np.ndarray, c_unit: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw one snapshot's loads and PV outputs.

        Args:
            x_unit: Unit design point (PV sizes).
            c_unit: Unit coefficient (load growth).
            rng: Call RNG.

        Returns:
            Tuple (load_p, load_q, pv_p) in physical units.
        """
        sizes, c_phys = self.to_physical(x_unit, c_unit)
        mult = np.exp(
            SIGMA_LN * rng.standard_normal(self.base_p.size) - SIGMA_LN**2 / 2.0
        )
        pv_frac = rng.uniform(*PV_OUTPUT_RANGE, size=N_PV_PLANTS)
        return (
            self.base_p * c_phys * mult,
            self.base_q * c_phys * mult,
            sizes * pv_frac,
        )

    def snapshot_margins(
        self, x_unit: np.ndarray, c_unit: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Raw margins of one snapshot (fast path).

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.
            rng: Call RNG (drives this snapshot's draws).

        Returns:
            Raw margins (g1, g2); (+3.0, +3.0) on non-convergence.
        """
        load_p, load_q, pv_p = self._snapshot_inputs(x_unit, c_unit, rng)
        out = self._solve_fast(load_p, load_q, pv_p)
        if out is None:
            return np.full(M_CHANNELS, SNAPSHOT_CRASH_MARGIN)
        vm, line_loading, trafo_loading = out
        loading = max(float(line_loading.max()), float(trafo_loading.max()))
        return np.array(
            [
                float(np.abs(vm - 1.0).max()) - V_DEV_LIMIT,
                (loading - 100.0) / 100.0,
            ]
        )

    def snapshot_margins_reference(
        self, x_unit: np.ndarray, c_unit: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Raw margins of one snapshot via ``pp.runpp`` (verification path).

        Draws the identical snapshot inputs as :meth:`snapshot_margins`
        given the same RNG state.

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.
            rng: Call RNG.

        Returns:
            Raw margins (g1, g2); (+3.0, +3.0) on non-convergence.
        """
        load_p, load_q, pv_p = self._snapshot_inputs(x_unit, c_unit, rng)
        net = self.net
        net.load["p_mw"] = load_p
        net.load["q_mvar"] = load_q
        for idx, p in zip(self.pv_idx, pv_p, strict=True):
            net.sgen.at[idx, "p_mw"] = p
        try:
            self._pp.runpp(net, **SOLVER_SETTINGS)
        except Exception:
            return np.full(M_CHANNELS, SNAPSHOT_CRASH_MARGIN)
        vm = net.res_bus.vm_pu.to_numpy()
        loading = max(
            float(net.res_line.loading_percent.max()),
            float(net.res_trafo.loading_percent.max()),
        )
        return np.array(
            [
                float(np.abs(vm - 1.0).max()) - V_DEV_LIMIT,
                (loading - 100.0) / 100.0,
            ]
        )

    def mean_margins(
        self,
        x_unit: np.ndarray,
        c_unit: float,
        k_snapshots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Mean raw margins over K fresh snapshots from the call RNG.

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.
            k_snapshots: Number of snapshots K.
            rng: Call RNG.

        Returns:
            Mean margins, shape (2,).
        """
        acc = np.zeros(M_CHANNELS)
        for _ in range(k_snapshots):
            acc += self.snapshot_margins(x_unit, c_unit, rng)
        return acc / k_snapshots

    def mean_feasible(
        self,
        x_unit: np.ndarray,
        c_unit: float,
        k_snapshots: int,
        rng: np.random.Generator,
    ) -> bool:
        """The K-snapshot-mean feasibility predicate.

        Args:
            x_unit: Unit design point, shape (4,).
            c_unit: Unit coefficient.
            k_snapshots: Number of snapshots K.
            rng: Call RNG.

        Returns:
            True when every mean margin is <= 0.
        """
        return bool((self.mean_margins(x_unit, c_unit, k_snapshots, rng) <= 0.0).all())


def verify_fast_path(
    oracle: HostingCapacityOracle,
    n_samples: int = 300,
    seed: int = 7001,
) -> dict:
    """Exact-equivalence verification of the fast path against pp.runpp.

    Samples snapshots across the design box and an extended load range
    (including collapse-prone points), compares convergence flags, and
    bounds the margin disagreement on converged snapshots.

    Args:
        oracle: Hosting-capacity raw oracle.
        n_samples: Number of verification snapshots.
        seed: Verification RNG seed.

    Returns:
        Dict with flag mismatches, the max margin deviation, and counts.
    """
    rng = np.random.default_rng(seed)
    n_flag_mismatch = 0
    n_nonconverged = 0
    max_margin_dev = 0.0
    for i in range(n_samples):
        x = rng.uniform(size=X_DIM)
        # Every third point stresses the collapse boundary beyond c = 1.
        c = rng.uniform(0.0, 2.5) if i % 3 == 0 else rng.uniform(0.0, 1.0)
        state = rng.bit_generator.state
        g_fast = oracle.snapshot_margins(x, c, rng)
        rng.bit_generator.state = state
        g_ref = oracle.snapshot_margins_reference(x, c, rng)
        fast_crashed = bool((g_fast == SNAPSHOT_CRASH_MARGIN).all())
        ref_crashed = bool((g_ref == SNAPSHOT_CRASH_MARGIN).all())
        if fast_crashed != ref_crashed:
            n_flag_mismatch += 1
            continue
        if fast_crashed:
            n_nonconverged += 1
            continue
        max_margin_dev = max(max_margin_dev, float(np.abs(g_fast - g_ref).max()))
    return {
        "n_samples": n_samples,
        "seed": seed,
        "n_convergence_flag_mismatches": n_flag_mismatch,
        "n_nonconverged": n_nonconverged,
        "max_margin_deviation": max_margin_dev,
        "ok": n_flag_mismatch == 0 and max_margin_dev < 1.0e-8,
    }


class _MeanPredicate:
    """K-snapshot-mean predicate with a rebindable per-call RNG."""

    def __init__(self, oracle: HostingCapacityOracle, k_snapshots: int) -> None:
        """Bind the raw oracle and the noise knob.

        Args:
            oracle: Hosting-capacity raw oracle.
            k_snapshots: Noise knob K.
        """
        self.oracle = oracle
        self.k_snapshots = k_snapshots
        self.rng: np.random.Generator | None = None

    def __call__(self, x_unit: np.ndarray, c_unit: float) -> bool:
        """Evaluate the predicate with K fresh snapshots from the call RNG."""
        if self.rng is None:
            raise ValueError("hosting-capacity predicate requires a call RNG")
        return self.oracle.mean_feasible(x_unit, c_unit, self.k_snapshots, self.rng)


@dataclass
class HostingCapacityEnvironment:
    """Frozen hosting-capacity environment behind the portfolio protocol.

    Attributes:
        k_snapshots: Noise knob K (snapshots per predicate
            evaluation); the plan's noise knob semantics.
        pv_buses: Frozen PV plant bus list.
        scaler_scale: Per-channel standardization scale IQR / 1.349, (2,).
        B: Frozen inflated joint-RKHS norm bounds, shape (2,).
        B_slice: Frozen inflated per-slice norm bounds, shape (2,).
        R_by_knob: Frozen noise scale per knob K (standardized channels).
        v_min: Frozen viability level.
        c_star: Unit-coefficient truth threshold.
        beta_hat: Fitted margin exponent (contraction diagnostic).
        v_grid: Frozen truth volume curve on the unit report grid.
        v_grid_c: Unit coefficients of the frozen volume curve.
        window_oracle: Lazily built window oracle (one per process).
    """

    k_snapshots: int
    pv_buses: list[int]
    prefix_only: bool
    prefix_no_window_report_grid_step: float | None
    scaler_scale: np.ndarray
    B: np.ndarray
    B_slice: np.ndarray
    R_by_knob: dict[int, float]
    v_min: float | None
    c_star: float | None
    beta_hat: float | None
    v_grid: np.ndarray | None
    v_grid_c: np.ndarray | None
    window_oracle: WindowMarginOracle | None = field(default=None, repr=False)
    _predicate: _MeanPredicate | None = field(default=None, repr=False)

    x_dim = X_DIM
    x_bounds = (
        PV_SIZE_RANGE_MW,
        PV_SIZE_RANGE_MW,
        PV_SIZE_RANGE_MW,
        PV_SIZE_RANGE_MW,
    )
    c_range = C_RANGE

    @property
    def m(self) -> int:
        """Number of window channels (hosting uses prefix m=1)."""
        return 1 if self.prefix_only else M_CHANNELS

    @property
    def noise_knob(self) -> int:
        """Noise knob: snapshots per predicate evaluation."""
        return self.k_snapshots

    def _oracle(self) -> WindowMarginOracle:
        """The per-instance window oracle, built on first use."""
        if self.window_oracle is None:
            raw = HostingCapacityOracle(pv_buses=self.pv_buses)
            self._predicate = _MeanPredicate(raw, self.k_snapshots)
            self.window_oracle = WindowMarginOracle(
                self._predicate,
                prefix_only=self.prefix_only,
                prefix_no_window_report_grid_step=(
                    self.prefix_no_window_report_grid_step if self.prefix_only else None
                ),
            )
        return self.window_oracle

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Standardized window margins at unit (x, c); stochastic.

        One oracle call runs the window extraction over the
        K-snapshot-mean predicate; every raw predicate evaluation inside
        the call consumes K fresh snapshots from the call RNG.

        Args:
            x: Unit design point, shape (4,).
            c: Unit coefficient.
            rng: Call RNG (required; drives all snapshot draws).
            sigma: Must be 0.0; the noise is intrinsic (knob K).

        Returns:
            Standardized channels (g_lo, g_hi) / scale, shape (2,),
            unclipped (exactly affine in c given the extracted window).

        Raises:
            ValueError: When sigma is nonzero or rng is missing.
        """
        if sigma != 0.0:
            raise ValueError(
                "hosting-capacity noise is intrinsic (knob K); sigma must be 0"
            )
        if rng is None:
            raise ValueError("hosting-capacity oracle requires a call RNG")
        oracle = self._oracle()
        self._predicate.rng = rng
        raw = oracle.raw_channels(np.asarray(x, dtype=float), float(c))
        return raw / self.scaler_scale

    def true_volume(self, c_unit: float) -> float | None:
        """Interpolate the frozen truth volume curve at a unit coefficient.

        Args:
            c_unit: Unit coefficient.

        Returns:
            Linearly interpolated true volume, or None before truth freeze.
        """
        if self.v_grid is None:
            return None
        return float(np.interp(c_unit, self.v_grid_c, self.v_grid))

    def window_counters(self) -> dict:
        """Sidecar accounting of the window oracle."""
        if self.window_oracle is None:
            return {
                "n_calls": 0,
                "n_raw_evals": 0,
                "raw_evals_per_call": 0.0,
                "n_multi_window": 0,
            }
        return self.window_oracle.counters()


def load_hosting_environment(
    path: str | Path, k_snapshots: int
) -> HostingCapacityEnvironment:
    """Load the frozen hosting-capacity environment from its JSON file.

    Args:
        path: Path to ``configs/hosting_capacity_frozen_calibration.json``.
        k_snapshots: Noise knob K for this run cell (from the run config).

    Returns:
        HostingCapacityEnvironment with frozen scalers, bounds, and truth.

    Raises:
        ValueError: When the frozen registration disagrees with the module
            constants, or K is not a knob with a frozen R.
    """
    payload = json.loads(Path(path).read_text())
    env_reg = payload["environment"]
    expected = {
        "network": NETWORK_NAME,
        "c_range": list(C_RANGE),
        "pv_size_range_mw": list(PV_SIZE_RANGE_MW),
        "sigma_ln": SIGMA_LN,
        "oracle_representation": "window-margin over the K-snapshot-mean "
        "feasibility predicate",
    }
    for key, value in expected.items():
        if env_reg[key] != value:
            raise ValueError(
                f"frozen registration mismatch for {key}: "
                f"file has {env_reg[key]!r}, module has {value!r}"
            )
    prefix_only = bool(env_reg.get("prefix_only", False))
    expected_m = 1 if prefix_only else M_CHANNELS
    if env_reg["m"] != expected_m:
        raise ValueError(
            f"environment m={env_reg['m']} disagrees with prefix_only={prefix_only}"
        )
    prefix_step = env_reg.get("prefix_no_window_report_grid_step")
    if prefix_only and prefix_step != PREFIX_NO_WINDOW_REPORT_GRID_STEP:
        raise ValueError(
            "P5 prefix no-window clamp mismatch: "
            f"file has {prefix_step!r}, expected "
            f"{PREFIX_NO_WINDOW_REPORT_GRID_STEP!r}"
        )
    cal = payload["calibration"]
    noise = payload["noise_scale"]
    R_by_knob = {int(k): float(v["R"]) for k, v in noise["by_knob"].items()}
    if int(k_snapshots) not in R_by_knob:
        raise ValueError(
            f"noise knob K={k_snapshots} has no frozen R; registered knobs: "
            f"{sorted(R_by_knob)}"
        )
    truth = payload.get("truth") or {}
    return HostingCapacityEnvironment(
        k_snapshots=int(k_snapshots),
        pv_buses=[int(b) for b in env_reg["pv_buses"]],
        prefix_only=prefix_only,
        prefix_no_window_report_grid_step=(
            float(prefix_step) if prefix_step is not None else None
        ),
        scaler_scale=np.asarray(cal["scaler_scale"], dtype=float),
        B=np.asarray(cal["B"], dtype=float),
        B_slice=np.asarray(cal["B_slice"], dtype=float),
        R_by_knob=R_by_knob,
        v_min=truth.get("v_min"),
        c_star=truth.get("c_star_unit"),
        beta_hat=truth.get("beta_hat"),
        v_grid=(
            np.asarray(truth["v_grid"], dtype=float) if "v_grid" in truth else None
        ),
        v_grid_c=(
            np.asarray(truth["v_grid_c"], dtype=float) if "v_grid_c" in truth else None
        ),
    )
