"""Covasim epidemic-policy study (Appendix C.5).

Covasim 3.x (pinned below 4) with its default random-network population,
``pop_size`` 20,000, 120 days, ``pop_infected`` 20, waning immunity off.

Design x (unit cube, budget-share coordinates; every point is affordable):

- u1: daily test probability of symptomatic people, ``t = 0.3 u1``
  (``cv.test_prob(symp_prob=t)``), cost ``100 t``;
- u2: share of the affordable tracing, ``r = u2 min(0.9, (40 - 100 t) / 20)``
  (``cv.contact_tracing(trace_probs=r)``), cost ``20 r``;
- u3: share of the remaining budget spent on distancing; distancing (beta
  halved, ``cv.change_beta(d, 0.5)``) starts on day
  ``d = 60 - floor(u3 (40 - 100 t - 20 r))``, cost ``60 - d``.

The map sends the cube onto the policies whose cost ``100 t + 20 r + (60 - d)``
is at most 40. Stress c: transmissibility multiplier in [0.8, 4.0] on
Covasim's default beta 0.016 (unit c in [0, 1]).

One simulation gives the margin ``g = (peak n_critical - cap) / cap`` with
``cap = 0.3 percent of the population`` (60 people). The pass/fail check is
the mean of ``g`` over K independent simulations being <= 0; every simulation
draws its Covasim ``rand_seed`` from the call RNG.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brink.window_margins import WindowMarginOracle

os.environ.setdefault("COVASIM_VERBOSE", "0")  # no license banner per worker
import covasim as cv  # noqa: E402

# Every cv.Sim records git metadata through sciris.gitinfo. In a git worktree
# (.git is a file) that falls back to GitPython and starts a `git cat-file`
# subprocess per run; one hung for 22 min and stalled a run. The metadata never
# enters the simulation, so it is stubbed.
cv.misc.git_info = lambda *args, **kwargs: {"covasim": {"version": cv.__version__}}

POP_SIZE = 20_000
N_DAYS = 120
POP_INFECTED = 20
BASE_BETA = 0.016  # Covasim 3.x default per-contact transmissibility
C_RANGE = (0.8, 4.0)  # transmissibility multiplier on beta
TEST_MAX = 0.3
TRACE_MAX = 0.9
START_DAY_MAX = 60
COST_CAP = 40.0
TEST_COST = 100.0  # per unit daily test probability
TRACE_COST = 20.0  # per unit trace probability
DISTANCING_FACTOR = 0.5
CAPACITY_FRACTION = 0.003
X_DIM = 3
PREFIX_NO_WINDOW_STEP = 1.0 / 200  # clamp: one report-grid step (grid 201)


def to_policy(x_unit: np.ndarray) -> tuple[float, float, int]:
    """Budget-share coordinates to the physical policy.

    Args:
        x_unit: (u1, u2, u3) in the unit cube.

    Returns:
        (test probability, trace probability, distancing start day).
    """
    u1, u2, u3 = (float(v) for v in np.clip(x_unit, 0.0, 1.0))
    t = TEST_MAX * u1
    left = COST_CAP - TEST_COST * t
    r = u2 * min(TRACE_MAX, left / TRACE_COST)
    left -= TRACE_COST * r
    d = START_DAY_MAX - int(np.floor(u3 * left + 1e-9))  # whole days, never over cap
    return t, r, d


def policy_cost(t: float, r: float, d: int) -> float:
    """Cost of a policy (the cap is 40)."""
    return TEST_COST * t + TRACE_COST * r + (START_DAY_MAX - d)


def beta_multiplier(c_unit: float) -> float:
    """Unit coefficient to the transmissibility multiplier in [0.8, 2.5]."""
    return C_RANGE[0] + (C_RANGE[1] - C_RANGE[0]) * float(c_unit)


def capacity() -> float:
    """Critical-care capacity: 0.3 percent of the population."""
    return CAPACITY_FRACTION * POP_SIZE


def simulate(x_unit: np.ndarray, c_unit: float, rand_seed: int) -> dict:
    """One Covasim run of the policy at the unit coefficient.

    Args:
        x_unit: Unit design point.
        c_unit: Unit coefficient.
        rand_seed: Covasim random seed.

    Returns:
        Dict with the peak critical count, the margin, and cumulative
        infections and tests.
    """
    t, r, d = to_policy(x_unit)
    sim = cv.Sim(
        pop_size=POP_SIZE, pop_infected=POP_INFECTED, n_days=N_DAYS,
        beta=BASE_BETA * beta_multiplier(c_unit), rand_seed=int(rand_seed),
        use_waning=False, verbose=0,
        interventions=[
            cv.test_prob(symp_prob=t),
            cv.contact_tracing(trace_probs=r),
            cv.change_beta(days=d, changes=DISTANCING_FACTOR),
        ],
    )  # fmt: skip
    sim.run()
    peak = float(sim.results["n_critical"].values.max())
    return {
        "peak_critical": peak,
        "margin": (peak - capacity()) / capacity(),
        "cum_infections": float(sim.results["cum_infections"].values[-1]),
        "cum_tests": float(sim.results["cum_tests"].values[-1]),
    }


def draw_seed(rng: np.random.Generator) -> int:
    """A Covasim seed from a call RNG."""
    return int(rng.integers(1, 2**31 - 1))


def margins(x_unit: np.ndarray, c_unit: float, k: int, rng) -> np.ndarray:
    """Margins of K independent runs, seeds drawn from the call RNG."""
    return np.array(
        [simulate(x_unit, c_unit, draw_seed(rng))["margin"] for _ in range(int(k))]
    )


def mean_feasible(x_unit: np.ndarray, c_unit: float, k: int, rng) -> bool:
    """Raw predicate: the K-run mean margin is <= 0."""
    return bool(margins(x_unit, c_unit, k, rng).mean() <= 0.0)


def window_oracle(k: int, rng_holder: list) -> WindowMarginOracle:
    """Upper-window oracle (m = 1) over the K-run predicate.

    Args:
        k: Runs per predicate evaluation (the noise knob).
        rng_holder: One-element list holding the current call RNG; the
            caller rebinds ``rng_holder[0]`` before each call.

    Returns:
        WindowMarginOracle with m = 1 (death channel only).
    """
    return WindowMarginOracle(
        lambda x, c: mean_feasible(x, c, k, rng_holder[0]),
        prefix_only=True,
        prefix_no_window_report_grid_step=PREFIX_NO_WINDOW_STEP,
    )


@dataclass
class CovasimPolicyEnvironment:
    """Frozen S5 environment behind the existence-arm protocol.

    Attributes:
        k_runs: Runs per raw predicate evaluation (noise knob K).
        scaler_scale: Window-channel scale (IQR / 1.349), shape (1,).
        B: Frozen fitted norm bound, shape (1,).
        R: Frozen noise scale at this K (standardized channel).
        lengthscale: Frozen RBF lengthscale.
    """

    k_runs: int
    scaler_scale: np.ndarray
    B: np.ndarray
    R: float
    lengthscale: float
    x_dim: int = X_DIM
    m: int = 1
    _rng: list = field(default_factory=lambda: [None], repr=False)
    _oracle: WindowMarginOracle | None = field(default=None, repr=False)

    def oracle(self, x, c, rng=None, sigma: float = 0.0) -> np.ndarray:
        """Standardized window death channel at unit (x, c).

        Args:
            x: Unit design point.
            c: Unit coefficient.
            rng: Call RNG (required; draws every run's seed).
            sigma: Must be 0 (the noise is intrinsic).

        Returns:
            Array (g_hi / scale,), exactly affine in c.

        Raises:
            ValueError: When sigma is nonzero or rng is missing.
        """
        if sigma != 0.0 or rng is None:
            raise ValueError("Covasim noise is intrinsic: pass rng and sigma = 0")
        if self._oracle is None:
            self._oracle = window_oracle(self.k_runs, self._rng)
        self._rng[0] = rng
        return self._oracle.raw_channels(np.asarray(x, float), float(c)) / (
            self.scaler_scale
        )

    def window_counters(self) -> dict:
        """Cumulative window-oracle accounting (raw evaluations per call)."""
        if self._oracle is None:
            return {"n_calls": 0, "n_raw_evals": 0, "n_multi_window": 0}
        return self._oracle.counters()


def load_environment(path: str | Path, k_runs: int) -> CovasimPolicyEnvironment:
    """The frozen S5 environment from its calibration JSON.

    Args:
        path: ``configs/covasim_policy_frozen_calibration.json``.
        k_runs: Noise knob K of this run cell.

    Returns:
        CovasimPolicyEnvironment.

    Raises:
        ValueError: When K has no frozen R or the file disagrees with the
            module constants.
    """
    payload = json.loads(Path(path).read_text())
    if payload["environment"] != environment_registration():
        raise ValueError("frozen S5 environment disagrees with the module constants")
    cal, by_k = payload["calibration"], payload["noise_scale"]["by_knob"]
    if str(k_runs) not in by_k:
        raise ValueError(f"K = {k_runs} has no frozen R; knobs {sorted(by_k)}")
    return CovasimPolicyEnvironment(
        k_runs=int(k_runs), scaler_scale=np.asarray(cal["scaler_scale"], float),
        B=np.asarray(cal["B"], float), R=float(by_k[str(k_runs)]["R"]),
        lengthscale=float(cal["lengthscale"]),
    )  # fmt: skip


def environment_registration() -> dict:
    """The environment constants recorded in (and checked against) the JSON."""
    return {
        "covasim": cv.__version__, "pop_size": POP_SIZE, "n_days": N_DAYS,
        "pop_infected": POP_INFECTED, "pop_type": "random", "use_waning": False,
        "base_beta": BASE_BETA, "c_range": list(C_RANGE), "test_max": TEST_MAX,
        "trace_max": TRACE_MAX, "start_day_max": START_DAY_MAX, "cost_cap": COST_CAP,
        "distancing_factor": DISTANCING_FACTOR,
        "capacity_fraction": CAPACITY_FRACTION,
        "x_map": "budget-share coordinates (covasim_policy.to_policy)",
        "oracle": "P5 prefix window channel over the K-run mean-margin predicate",
        "prefix_no_window_step": PREFIX_NO_WINDOW_STEP,
    }  # fmt: skip
