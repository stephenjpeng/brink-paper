"""Hopper study: co-design under a fixed pretrained policy (Appendix C.5).

Gymnasium Hopper-v5 with an XML-modified morphology, driven by the pretrained
SAC Hopper-v3 policy of ``sb3/sac-Hopper-v3`` (Hugging Face; fetched by
``scripts/hopper_setup.py download-policy``). Design x (unit box, 3 dims):
leg length scale (thigh and leg jointly) in [0.6, 1.4], foot length scale in
[0.6, 1.4], torso mass scale in [0.6, 1.6]. Stress c: payload in kg on the
torso, on ``C_RANGE_KG`` mapped to unit c in [0, 1].

One raw evaluation = one deterministic 1000-step rollout from
``reset(seed=0)`` with termination disabled. Raw margins (m = 2), feasible
iff both <= 0:

    g1 = 350 - 100 * x_final          (forward distance at least 3.5 m)
    g2 = 10 * (0.7 - min torso height) (torso never below 0.7 m)

A rollout that raises a MuJoCo error or produces a non-finite state counts as
a crash, which is infeasible.

Policy loading runs no pickled code: the SB3 zip is opened with ``zipfile``
and only ``policy.pth`` is read, with ``torch.load(weights_only=True)``. The
actor's deterministic action is tanh(mu(obs)), evaluated in float32 numpy.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv

from brink.window_margins import WindowMarginOracle

logger = logging.getLogger(__name__)

# Constants.
X_BOUNDS = ((0.6, 1.4), (0.6, 1.4), (0.6, 1.6))
X_DIM = 3
C_RANGE_KG = (0.0, 20.0)
N_STEPS = 1000
RESET_SEED = 0
DISTANCE_MIN_M = 3.5
HEIGHT_MIN_M = 0.7
# Window representation: prefix m = 1 on the death channel
# g_hi = c - c_off(x) with the no-window clamp c_off = -REPORT_GRID_STEP.
# theta_exists = sup_x c_off(x) does not depend on the onset channel; the m = 2
# calibration (B = 2529, 1527 from the 1.25 no-window cliff) is on record.
PREFIX_ONLY = True
REPORT_GRID_STEP = 0.005  # unit c; 201-point report grid
POLICY_ZIP = Path("data/raw/sb3_sac_hopper_v3/sac-Hopper-v3.zip")
POLICY_ZIP_SHA256 = "79157a5e47acb2a89d272891d3573040ab2638f592f440989365638084a3fd54"

# Stock Hopper-v5 geometry (gymnasium assets/hopper.xml), metres.
_THIGH_LEN, _LEG_LEN = 0.45, 0.5
_ANKLE_Z = 0.1  # foot geom centre height; unchanged by the scales
_TORSO_HALF = 0.2
_FOOT_HALF, _FOOT_OFFSET = 0.195, 0.065  # capsule half-length, centre past ankle
_ANKLE_X_IN_FOOT = -0.13
_DENSITY = 1000.0  # MuJoCo default geom density


def to_physical(x_unit: np.ndarray, c_unit: float, c_range=C_RANGE_KG) -> tuple:
    """Map a unit design and unit coefficient to physical scales and kg.

    Args:
        x_unit: Unit design point, shape (3,).
        c_unit: Unit coefficient.
        c_range: Physical payload range in kg.

    Returns:
        Tuple (leg_scale, foot_scale, torso_mass_scale, payload_kg).
    """
    x = np.asarray(x_unit, dtype=float)
    lo = np.array([b[0] for b in X_BOUNDS])
    hi = np.array([b[1] for b in X_BOUNDS])
    leg, foot, torso = lo + (hi - lo) * x
    payload = c_range[0] + (c_range[1] - c_range[0]) * float(c_unit)
    return float(leg), float(foot), float(torso), float(payload)


def hopper_xml(leg: float, foot: float, torso_mass: float, payload_kg: float) -> str:
    """Stock Hopper-v5 XML with the S4 morphology and payload applied.

    The thigh and leg capsules are scaled by ``leg`` with every joint and
    body position moved so the chain stays connected and the foot keeps its
    stock height; the torso height (and the ``rootz`` reference, so qpos[1]
    stays the absolute torso height) follows. The foot capsule half-length
    and its offset past the ankle scale by ``foot``. The torso geom density
    scales by ``torso_mass``. The payload is a non-colliding sphere at the
    torso centre with mass ``payload_kg``.

    Args:
        leg: Leg length scale.
        foot: Foot length scale.
        torso_mass: Torso mass scale.
        payload_kg: Payload mass in kg (>= 0).

    Returns:
        The modified MJCF document as a string.
    """
    root = ET.parse(_stock_xml_path()).getroot()
    body = {b.get("name"): b for b in root.iter("body")}
    geom = {g.get("name"): g for g in root.iter("geom")}
    joint = {j.get("name"): j for j in root.iter("joint")}

    def fmt(*v: float) -> str:
        return " ".join(repr(float(a)) for a in v)

    thigh_len, leg_len = _THIGH_LEN * leg, _LEG_LEN * leg
    torso_z = _ANKLE_Z + leg_len + thigh_len + _TORSO_HALF
    body["torso"].set("pos", fmt(0, 0, torso_z))
    for name in ("rootx", "rootz"):
        joint[name].set("pos", fmt(0, 0, -torso_z))
    joint["rootz"].set("ref", repr(torso_z))
    geom["thigh_geom"].set("pos", fmt(0, 0, -thigh_len / 2))
    geom["thigh_geom"].set("size", fmt(0.05, thigh_len / 2))
    body["leg"].set("pos", fmt(0, 0, -(thigh_len + leg_len / 2)))
    joint["leg_joint"].set("pos", fmt(0, 0, leg_len / 2))
    geom["leg_geom"].set("size", fmt(0.04, leg_len / 2))
    body["foot"].set("pos", fmt(-_ANKLE_X_IN_FOOT, 0, -(leg_len / 2 + _ANKLE_Z)))
    geom["foot_geom"].set("pos", fmt(_ANKLE_X_IN_FOOT + _FOOT_OFFSET * foot, 0, 0.1))
    geom["foot_geom"].set("size", fmt(0.06, _FOOT_HALF * foot))
    geom["torso_geom"].set("density", repr(_DENSITY * torso_mass))
    ET.SubElement(
        body["torso"],
        "geom",
        {
            "name": "payload",
            "type": "sphere",
            "size": "0.05",
            "pos": "0 0 0",
            "mass": repr(float(payload_kg)),
            "contype": "0",
            "conaffinity": "0",
        },
    )
    return ET.tostring(root, encoding="unicode")


def _stock_xml_path() -> Path:
    """Path of the stock Hopper-v5 MJCF shipped with gymnasium."""
    import gymnasium.envs.mujoco as gm

    return Path(gm.__file__).parent / "assets" / "hopper.xml"


class _HopperFromString(HopperEnv):
    """Hopper-v5 whose MuJoCo model compiles from an MJCF string."""

    def __init__(self, xml: str, **kwargs) -> None:
        self._xml = xml
        super().__init__(**kwargs)

    def _initialize_simulation(self):
        model = mujoco.MjModel.from_xml_string(self._xml)
        return model, mujoco.MjData(model)


def load_actor(zip_path: str | Path = POLICY_ZIP) -> dict[str, np.ndarray]:
    """Actor weights of the SB3 SAC policy, read without unpickling.

    Args:
        zip_path: SB3 model zip (downloaded, SHA-256 in ``docs/data.md``).

    Returns:
        Dict of float32 arrays: W0, b0, W1, b1, Wmu, bmu.

    Raises:
        FileNotFoundError: When the policy zip is missing (download it per
            ``docs/data.md``; no stand-in policy is substituted).
    """
    import torch

    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: download sb3/sac-Hopper-v3 per docs/data.md"
        )
    with zipfile.ZipFile(path) as z:
        sd = torch.load(
            io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True
        )
    keys = {
        "W0": "actor.latent_pi.0.weight",
        "b0": "actor.latent_pi.0.bias",
        "W1": "actor.latent_pi.2.weight",
        "b1": "actor.latent_pi.2.bias",
        "Wmu": "actor.mu.weight",
        "bmu": "actor.mu.bias",
    }
    return {k: sd[v].numpy().astype(np.float32) for k, v in keys.items()}


def actor_action(w: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    """Deterministic SAC action tanh(mu(obs)) in float32.

    Args:
        w: Weights from :func:`load_actor`.
        obs: Observation, shape (11,) or (n, 11).

    Returns:
        Action(s) in [-1, 1], shape (3,) or (n, 3).
    """
    h = np.maximum(np.asarray(obs, dtype=np.float32) @ w["W0"].T + w["b0"], 0.0)
    h = np.maximum(h @ w["W1"].T + w["b1"], 0.0)
    return np.tanh(h @ w["Wmu"].T + w["bmu"])


@dataclass(frozen=True)
class Rollout:
    """Outcome of one rollout: final x position, minimum torso height, crash."""

    x_final: float
    min_height: float
    crashed: bool

    def margins(self) -> np.ndarray:
        """Raw margins (g1, g2); a crash returns +inf in both."""
        if self.crashed:
            return np.array([np.inf, np.inf])
        return np.array(
            [
                100.0 * (DISTANCE_MIN_M - self.x_final),
                10.0 * (HEIGHT_MIN_M - self.min_height),
            ]
        )


def rollout(
    xml: str,
    w: dict[str, np.ndarray],
    n_steps: int = N_STEPS,
    reset_seed: int = RESET_SEED,
) -> Rollout:
    """One deterministic Hopper-v5 rollout of the frozen policy.

    Args:
        xml: MJCF string (see :func:`hopper_xml`).
        w: Actor weights.
        n_steps: Environment steps (frame skip 4 each).
        reset_seed: Seed of ``env.reset`` (Hopper-v5 adds uniform noise of
            scale 5e-3 to the initial state).

    Returns:
        Rollout outcome; crashed when MuJoCo raises, warns of instability,
        or the state turns non-finite.
    """
    try:
        env = _HopperFromString(xml, terminate_when_unhealthy=False)
        obs, _ = env.reset(seed=reset_seed)
        min_z = float(env.data.qpos[1])
        for _ in range(n_steps):
            obs, *_ = env.step(actor_action(w, obs))
            min_z = min(min_z, float(env.data.qpos[1]))
        bad = env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
        x_final = float(env.data.qpos[0])
        env.close()
    except (mujoco.FatalError, ValueError) as err:
        logger.debug("rollout crashed: %s", err)
        return Rollout(float("nan"), float("nan"), True)
    if bad or not np.isfinite(obs).all() or not np.isfinite(x_final):
        return Rollout(x_final, min_z, True)
    return Rollout(x_final, min_z, False)


class HopperOracle:
    """Raw S4 oracle on unit coordinates; loads the actor once per instance.

    Args:
        zip_path: Policy zip path.
        c_range: Physical payload range (kg) mapped to unit c.
    """

    def __init__(self, zip_path: str | Path = POLICY_ZIP, c_range=C_RANGE_KG):
        """Load the frozen actor."""
        self.w = load_actor(zip_path)
        self.c_range = tuple(float(v) for v in c_range)
        self.n_rollouts = 0

    def rollout(
        self, x_unit: np.ndarray, c_unit: float, reset_seed: int = RESET_SEED
    ) -> Rollout:
        """Rollout at a unit design and unit coefficient."""
        self.n_rollouts += 1
        phys = to_physical(x_unit, c_unit, self.c_range)
        return rollout(hopper_xml(*phys), self.w, reset_seed=reset_seed)

    def margins(self, x_unit: np.ndarray, c_unit: float) -> np.ndarray:
        """Raw margins (g1, g2) at unit (x, c); +inf on crash."""
        return self.rollout(x_unit, c_unit).margins()

    def feasible(self, x_unit: np.ndarray, c_unit: float) -> bool:
        """Raw feasibility predicate (crash = infeasible)."""
        return bool(np.all(self.margins(x_unit, c_unit) <= 0.0))


def make_window_oracle(c_range=C_RANGE_KG) -> WindowMarginOracle:
    """S4 window oracle: prefix m = 1 with the no-window clamp.

    Args:
        c_range: Physical payload range (kg) mapped to unit c.

    Returns:
        WindowMarginOracle over the raw rollout predicate.
    """
    return WindowMarginOracle(
        HopperOracle(c_range=c_range).feasible,
        prefix_only=PREFIX_ONLY,
        prefix_no_window_report_grid_step=REPORT_GRID_STEP,
    )


@dataclass
class HopperEnvironment:
    """S4 window-margin environment behind the existence-arm interface.

    Attributes:
        scaler_scale: Per-channel standardization scale of the window
            channels (frozen at calibration).
        c_range: Physical payload range in kg.
        window_oracle: Lazily built window oracle (one actor per process).
    """

    scaler_scale: np.ndarray
    c_range: tuple[float, float] = C_RANGE_KG
    window_oracle: WindowMarginOracle | None = field(default=None, repr=False)

    x_dim = X_DIM
    m = 1

    def _oracle(self) -> WindowMarginOracle:
        if self.window_oracle is None:
            self.window_oracle = make_window_oracle(self.c_range)
        return self.window_oracle

    def oracle(
        self,
        x: np.ndarray,
        c: float,
        rng: np.random.Generator | None = None,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Standardized window channels at unit (x, c); deterministic.

        Raises:
            ValueError: When sigma is nonzero.
        """
        if sigma != 0.0:
            raise ValueError("the Hopper oracle is deterministic; sigma must be 0")
        raw = self._oracle().raw_channels(np.asarray(x, dtype=float), float(c))
        return raw / self.scaler_scale

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
