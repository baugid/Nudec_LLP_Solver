#!/usr/bin/env python3
"""
CLI runner for the neutrino-decoupling + LLP decay system.

Features:
  1) LLP parameters are supplied on the command line.
  2) Each run is assigned a unique integer id in [1, 10^6] that is not
     already present in the summary output file.
  3) A one-line summary is appended to the chosen summary output file.
  4) Distribution snapshots are written to <id>_nue-distr.txt on a fixed
     temperature grid.
  5) Thermodynamics snapshots are written to <id>_thermodynamics.txt on the
     same temperature grid.
  6) The summary output also includes
     ((a(T_start) * T_start) / (a(T_fin) * T_fin))^3 before N_eff.

Important note:
  The requested output columns are named n_nu_e/n_gamma, n_nu_mu/n_gamma,
  n_nu_tau/n_gamma. This script computes those as NUMBER-density ratios to
  the photon number density n_gamma = 2 zeta(3) / pi^2 * T^3.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from scipy import integrate
from scipy.integrate import solve_ivp

import Distributions
import Momentum_Grid
from Constants import *
import System_Nudecoupling as system_nudecoupling_module
from System_Nudecoupling import System_Nudec
try:
    from Thermal_QED_corrections import Thermal_QED_corrections_to_energy_density
except ModuleNotFoundError:
    from Thermodynamics.Thermal_QED_corrections import Thermal_QED_corrections_to_energy_density
from GlobalParameters import (
    SMComovingLimit,
    finalX,
    initialX,
    timeScaleFactorFile,
    useDecayProbabilities as USE_DECAY_PROBABILITIES_CFG,
    ifDebugging as IF_DEBUGGING_DEFAULT,
)


OUTPUT_ROOT_DEFAULT = Path("/eos/user/o/ovchynni/Traditional")
OUTPUT_FILE_BASENAME_DEFAULT = "output.txt"
RUN_ID_MIN = 1
RUN_ID_MAX = 10**6
ZETA3 = 1.2020569031595942854
DEFAULT_Z0_FOR_TSTART = 1.00003
DEFAULT_T_FINAL = 0.025
DEFAULT_MAX_X_END = 1.0e5
GRID_INJECTION_SAFETY = 2.0
THERMAL_COLLISION_U_MAX = 15.0
THERMAL_COLLISION_Y_LIMIT_CAP = 120.0
LLP_DOMINATION_GRID_THRESHOLD = 2.0
SUMMARY_HEADER = (
    "# id llp_mass llp_lifetime nbins llp_abundance llp_pionBranching "
    "llp_muonBranching llp_twoNuDecayE llp_twoNuDecayMu llp_twoNuDecayTau "
    "n_nu_e/n_gamma n_nu_mu/n_gamma n_nu_tau/n_gamma "
    "aT_ratio_cubed Neff\n"
)


def resolve_script_relative_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use True or False."
    )


def convert_Tstart_to_xstart(Tstart: float) -> float:
    if Tstart <= 0.0:
        raise ValueError("Tstart must be positive.")
    return float(DEFAULT_Z0_FOR_TSTART * me / Tstart)


def output_folder_name(value: str) -> str:
    cleaned = value.strip()
    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)

    if not cleaned:
        raise argparse.ArgumentTypeError("--output-folder must not be empty.")
    if cleaned in {".", ".."} or any(sep in cleaned for sep in separators):
        raise argparse.ArgumentTypeError(
            "--output-folder must be a simple folder name, not a path."
        )
    return cleaned


def build_output_dir(output_folder: str) -> Path:
    output_root = OUTPUT_ROOT_DEFAULT
    output_root.mkdir(parents=True, exist_ok=True)

    output_dir = output_root / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def estimate_thermal_collision_grid_limit(
    grid_stop_x: float,
    llp_mass: float,
    llp_abundance: float,
    llp_lifetime: float,
    T_start: float,
) -> float:
    """Extend the collision grid when entropy injection makes y = a p large."""
    rho_llp_start = llp_mass * llp_abundance
    rho_rad_start = (np.pi**2 / 30.0) * 10.75 * T_start**4

    if rho_rad_start <= 0.0 or rho_llp_start / rho_rad_start <= LLP_DOMINATION_GRID_THRESHOLD:
        return 0.0

    T_reheat = 0.7 / np.sqrt(llp_lifetime)
    z_reheat_est = grid_stop_x * T_reheat / me
    if z_reheat_est <= 0.0:
        return 0.0

    return float(min(THERMAL_COLLISION_Y_LIMIT_CAP, THERMAL_COLLISION_U_MAX * z_reheat_est))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLP neutrino-decoupling calculation and write summary/distribution outputs."
    )

    parser.add_argument("--llp-mass", type=float, default=1000.0, help="LLP mass in MeV")
    parser.add_argument("--llp-lifetime", type=float, default=1.0, help="LLP lifetime in s")
    parser.add_argument(
        "--llp-abundance",
        type=float,
        default=6.036,
        help="LLP number density at T = 5.11 MeV, in MeV^3",
    )
    parser.add_argument(
        "--llp-pion-branching",
        type=float,
        default=0.0,
        help="Average number of primary charged pions per LLP decay",
    )
    parser.add_argument(
        "--llp-muon-branching",
        type=float,
        default=0.0,
        help="Average number of primary muons per LLP decay",
    )
    parser.add_argument(
        "--llp-two-nu-decay-e",
        type=float,
        default=0.3333,
        help="Branching to two-nu decay producing electron flavor",
    )
    parser.add_argument(
        "--llp-two-nu-decay-mu",
        type=float,
        default=0.3333,
        help="Branching to two-nu decay producing muon flavor",
    )
    parser.add_argument(
        "--llp-two-nu-decay-tau",
        type=float,
        default=0.3334,
        help="Branching to two-nu decay producing tau flavor",
    )
    parser.add_argument(
        "--lifetime-factor",
        type=float,
        default=20.0,
        help="Injection cutoff multiplier: stop at lifetime_factor * llp_lifetime",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        required=True,
        help="Number of momentum grid points",
    )
    parser.add_argument(
        "--Tstart",
        type=float,
        default=DEFAULT_Z0_FOR_TSTART * me / initialX,
        help=(
            "Initial plasma temperature in MeV. Internally converted to x_start "
            f"using x_start = {DEFAULT_Z0_FOR_TSTART} * me / Tstart "
            f"(default chosen to reproduce GlobalParameters.py initialX={initialX})."
        ),
    )
    parser.add_argument(
        "--x-end",
        type=float,
        default=finalX,
        help=(
            "Initial safety ceiling for x. The run finalizes at --Tfinal; "
            f"if needed this ceiling is extended automatically (default: {finalX})."
        ),
    )
    parser.add_argument(
        "--max-x-end",
        type=float,
        default=DEFAULT_MAX_X_END,
        help=(
            "Maximum x ceiling allowed while extending the integration to --Tfinal "
            f"(default: {DEFAULT_MAX_X_END:g})."
        ),
    )
    parser.add_argument(
        "--Tfinal",
        type=float,
        default=DEFAULT_T_FINAL,
        help=f"Finalize the run when T_gamma reaches this MeV value (default: {DEFAULT_T_FINAL}).",
    )
    parser.add_argument(
        "--z0",
        type=float,
        default=1.00003,
        help="Initial z value",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.0,
        help="Initial cosmic time in s used in the state vector",
    )
    parser.add_argument(
        "--solver",
        default="RK23",
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        help="SciPy solve_ivp method",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance for solve_ivp")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for solve_ivp")
    parser.add_argument(
        "--no-interactions",
        type=str_to_bool,
        default=False,
        metavar="BOOL",
        help=(
            "Set to True to turn off nu-EM and nu-nu collision terms and keep only LLP decay injection. "
            "Example: --no-interactions True"
        ),
    )
    parser.add_argument(
        "--ifDebugging",
        type=str_to_bool,
        default=IF_DEBUGGING_DEFAULT,
        metavar="BOOL",
        help=(
            "Set to True to enable the nu-self debugging mode: compute the self-collision term a second time "
            "and subtract its energy moment from dz/dx only. Default comes from GlobalParameters.py. "
            "Example: --ifDebugging True"
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=output_folder_name,
        required=True,
        help="Name of the subfolder XXX under /eos/user/o/ovchynni/Traditional/XXX for all outputs.",
    )
    parser.add_argument(
        "--name-output",
        type=str,
        default=OUTPUT_FILE_BASENAME_DEFAULT,
        help="Summary output filename written inside /eos/user/o/ovchynni/Traditional/XXX (default: output.txt)",
    )
    parser.add_argument(
        "--no-decay-probabilities",
        action="store_true",
        help="Ignore decayProbs.csv and use unit decay probabilities",
    )

    args = parser.parse_args()

    if args.nbins < 3:
        parser.error("--nbins must be >= 3")
    if args.Tstart <= 0.0:
        parser.error("--Tstart must be > 0")
    if args.Tfinal <= 0.0:
        parser.error("--Tfinal must be > 0")
    if args.Tfinal >= args.Tstart:
        parser.error("--Tfinal must be smaller than --Tstart")
    if args.llp_lifetime <= 0.0:
        parser.error("--llp-lifetime must be > 0")
    if args.lifetime_factor <= 0.0:
        parser.error("--lifetime-factor must be > 0")
    if args.x_end <= 0.0:
        parser.error("--x-end must be > 0")
    if args.max_x_end <= args.x_end:
        parser.error("--max-x-end must be larger than --x-end")
    if os.path.isabs(args.name_output):
        parser.error("--name-output must be a filename, not an absolute path")
    if not args.name_output.strip():
        parser.error("--name-output must not be empty")

    args.x_start = convert_Tstart_to_xstart(args.Tstart)
    if args.x_end <= args.x_start:
        parser.error("--x-end must be larger than the x value implied by --Tstart")
    return args


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_existing_ids(path: str) -> set:
    ids = set()
    try:
        if not os.path.exists(path):
            return ids
    except OSError:
        return ids

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                first = stripped.split()[0]
                try:
                    ids.add(int(first))
                except ValueError:
                    continue
    except OSError:
        return ids
    return ids


def generate_unique_run_id(existing_ids: set) -> int:
    max_available = RUN_ID_MAX - RUN_ID_MIN + 1
    if len(existing_ids) >= max_available:
        raise RuntimeError("No unused run id left in the interval [1, 10^6].")

    while True:
        run_id = random.randint(RUN_ID_MIN, RUN_ID_MAX)
        if run_id not in existing_ids:
            return run_id


def initialize_output_file(path: str) -> None:
    ensure_parent_dir(path)

    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(SUMMARY_HEADER)
    except FileExistsError:
        return
    except OSError as create_error:
        if os.path.exists(path):
            return
        try:
            with open(path, "a", encoding="utf-8") as handle:
                if handle.tell() == 0:
                    handle.write(SUMMARY_HEADER)
            return
        except OSError:
            raise create_error


def build_stop_point(target_time_s: float, table_path: str) -> float:
    resolved_table_path = resolve_script_relative_path(table_path)
    x_time_data = np.loadtxt(resolved_table_path, delimiter=",")
    x_time_data = np.atleast_2d(x_time_data)
    if x_time_data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {resolved_table_path}.")

    times = x_time_data[:, 1]

    if target_time_s <= times[0]:
        return float(x_time_data[0, 0])
    if target_time_s >= times[-1]:
        return float(x_time_data[-1, 0])

    idx = np.searchsorted(times, target_time_s, side="right")
    x0, t0 = x_time_data[idx - 1, 0], x_time_data[idx - 1, 1]
    x1, t1 = x_time_data[idx, 0], x_time_data[idx, 1]
    return float(x0 + (target_time_s - t0) * (x1 - x0) / (t1 - t0))


def sm_time_at_x(x: float, x_time_data: np.ndarray) -> float:
    xs = x_time_data[:, 0]
    ts = x_time_data[:, 1]

    if x <= xs[0]:
        slope = (ts[1] - ts[0]) / (xs[1] - xs[0])
        return float(ts[0] + slope * (x - xs[0]))
    if x <= xs[-1]:
        return float(np.interp(x, xs, ts))

    return float(ts[-1] * (x / xs[-1]) ** 2)


def estimate_grid_stop_x(
    target_elapsed_time_s: float,
    table_path: str,
    x_start: float,
    x_end: float,
    llp_count: float,
    llp_lifetime: float,
    llp_mass: float,
    t_start: float,
) -> float:
    """
    Estimate the largest x reached while LLP injection is active.

    This uses the SM x-t table as a background clock, then rescales each
    interval by sqrt(1 + rho_LLP/rho_SM). It is only for choosing a safe
    momentum-grid range; the real evolution still uses the full ODE.
    """
    target_time_s = float(t_start) + float(target_elapsed_time_s)

    if target_elapsed_time_s <= 0.0:
        return float(x_start)
    if llp_lifetime <= 0.0 or llp_mass <= 0.0 or llp_count <= 0.0:
        return float(build_stop_point(target_time_s, table_path))

    resolved_table_path = resolve_script_relative_path(table_path)
    x_time_data = np.atleast_2d(np.loadtxt(resolved_table_path, delimiter=","))
    xs = x_time_data[:, 0]

    interior = xs[(x_start < xs) & (xs < x_end)]
    x_steps = np.concatenate(([x_start], interior, [x_end]))
    x_steps = np.unique(x_steps)
    if len(x_steps) < 2:
        return float(x_start)

    t_est = float(t_start)
    for left, right in zip(x_steps[:-1], x_steps[1:]):
        t_sm_left = sm_time_at_x(float(left), x_time_data)
        t_sm_right = sm_time_at_x(float(right), x_time_data)
        dt_sm = t_sm_right - t_sm_left
        if dt_sm <= 0.0:
            continue

        x_mid = 0.5 * (left + right)
        dtdx_sm = dt_sm / (right - left)
        H_sm = hbar / (x_mid * dtdx_sm)
        rho_sm = 3.0 * mpl**2 * H_sm**2 / (8.0 * np.pi)

        elapsed_t = max(0.0, t_est - t_start)
        rho_llp = llp_mass * llp_count * np.exp(-elapsed_t / llp_lifetime) * (me / x_mid) ** 3
        dt_est = dt_sm / np.sqrt(1.0 + max(0.0, rho_llp / rho_sm))

        if t_est + dt_est >= target_time_s:
            frac = (target_time_s - t_est) / dt_est
            x_cut = left + frac * (right - left)
            safety = x_start + 1.15 * (x_cut - x_start)
            return float(min(x_end, max(x_start, safety)))

        t_est += dt_est

    return float(x_end)


def make_decay_handler(use_decay_probabilities: bool):
    if not use_decay_probabilities:
        return lambda T: {"mu": 1.0, "pi": 1.0}

    dec_probabilities_path = resolve_script_relative_path("decayProbs.csv")
    dec_probabilities = np.loadtxt(dec_probabilities_path, delimiter=",")[::-1]
    dec_probabilities = np.atleast_2d(dec_probabilities)
    xs = dec_probabilities[:, 0]

    def decay_handler(T: float):
        idx = np.searchsorted(xs, T, side="right")

        if idx == 0:
            return {"mu": dec_probabilities[0, 1], "pi": dec_probabilities[0, 2]}
        if idx >= len(dec_probabilities):
            return {"mu": dec_probabilities[-1, 1], "pi": dec_probabilities[-1, 2]}

        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = dec_probabilities[idx - 1], dec_probabilities[idx]
        vals = y0 + (T - x0) * (y1 - y0) / (x1 - x0)
        return {"mu": vals[1], "pi": vals[2]}

    return decay_handler


def fermi_dirac_zero_mu(arg):
    return np.exp(-np.logaddexp(0.0, arg))


def calc_neff(state: np.ndarray) -> float:
    n = Momentum_Grid.n

    rho1 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[:n])
    rho2 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[n:2 * n])
    rho3 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[2 * n:3 * n])
    rho4 = np.sum(
        Momentum_Grid.gridWeights
        * Momentum_Grid.gridVals**3
        * fermi_dirac_zero_mu(Momentum_Grid.gridVals)
    )

    z = state[3 * n]
    return float(((11.0 / 4.0) ** (1.0 / 3.0) / z) ** 4 * (rho1 + rho2 + rho3) / rho4)


def photon_number_density(T: float) -> float:
    return 2.0 * ZETA3 / np.pi**2 * T**3


def neutrino_plus_antineutrino_number_density_from_state(state: np.ndarray, flavor_index: int, x: float) -> float:
    n = Momentum_Grid.n
    start = flavor_index * n
    stop = (flavor_index + 1) * n
    fvals = state[start:stop]

    prefactor = (me / x) ** 3 / np.pi**2
    integral = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**2 * fvals)
    return float(prefactor * integral)


def neutrino_plus_antineutrino_energy_density_from_state(state: np.ndarray, flavor_index: int, x: float) -> float:
    n = Momentum_Grid.n
    start = flavor_index * n
    stop = (flavor_index + 1) * n
    fvals = state[start:stop]

    prefactor = (me / x) ** 4 / np.pi**2
    integral = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * fvals)
    return float(prefactor * integral)


def electron_positron_energy_density_from_xz(x: float, z: float) -> float:
    y = z * np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    integrand = (
        2.0
        * y**2
        * np.sqrt(y**2 + x**2)
        / (np.exp(np.sqrt(y**2 + x**2) / z) + 1.0)
    )
    rho_e_bar = integrate.simpson(integrand, x=y) / np.pi**2
    return float((me / x) ** 4 * rho_e_bar)


def em_energy_density_from_state(state: np.ndarray, x: float) -> float:
    n = Momentum_Grid.n
    z = float(state[3 * n])
    T = z * me / x

    rho_gamma = np.pi**2 / 15.0 * T**4
    rho_e = electron_positron_energy_density_from_xz(x, z)
    rho_2_bar, rho_3_bar = Thermal_QED_corrections_to_energy_density(x, z)
    rho_qed = float((me / x) ** 4 * (rho_2_bar + rho_3_bar))

    return float(rho_gamma + rho_e + rho_qed)


def final_temperature_from_state(state: np.ndarray, x: float) -> float:
    return float(state[3 * Momentum_Grid.n] / x * me)


def aT_ratio_cubed_from_states(
    state_start: np.ndarray,
    x_start: float,
    state_fin: np.ndarray,
    x_fin: float,
) -> float:
    """
    Compute:
        ((a(T_start) * T_start) / (a(T_fin) * T_fin))^3

    In this code:
        x = a * me
        T = z * me / x
    so a*T = z, and this factor is equivalently (z_start / z_fin)^3.
    """
    T_start = final_temperature_from_state(state_start, x_start)
    T_fin = final_temperature_from_state(state_fin, x_fin)

    a_start = x_start / me
    a_fin = x_fin / me

    aT_start = a_start * T_start
    aT_fin = a_fin * T_fin

    return float((aT_start / aT_fin) ** 3)


def build_temperature_targets(T_start: float, T_fin: float) -> List[float]:
    fixed_grid = sorted(
        {
            4.75,
            4.5,
            4.0,
            3.75,
            3.5,
            3.25,
            3.0,
            2.75,
            2.5,
            2.25,
            2.0,
            1.75,
            1.5,
            1.25,
            1.0,
            0.95,
            0.9,
            0.85,
            0.8,
            0.75,
            0.7,
            0.65,
            0.6,
            0.55,
            0.5,
            0.45,
            0.4,
            0.35,
            0.3,
            0.25,
            0.2,
            0.15,
            0.1,
            0.08,
            0.07,
            0.06,
            0.05,
            0.04,
            0.03,
        },
        reverse=True,
    )

    targets = [T_start]
    for T in fixed_grid:
        if T_fin < T < T_start:
            targets.append(float(T))
    targets.append(T_fin)

    cleaned = []
    for T in targets:
        if not cleaned or abs(T - cleaned[-1]) > 1e-12:
            cleaned.append(float(T))
    return cleaned


def interpolate_state_at_temperature(
    xs: np.ndarray,
    ys: np.ndarray,
    temperatures: np.ndarray,
    target_T: float,
) -> Tuple[float, np.ndarray, float]:
    if target_T >= temperatures[0]:
        return float(xs[0]), ys[:, 0].copy(), float(temperatures[0])
    if target_T <= temperatures[-1]:
        return float(xs[-1]), ys[:, -1].copy(), float(temperatures[-1])

    for i in range(len(xs) - 1):
        T0 = temperatures[i]
        T1 = temperatures[i + 1]
        if (T0 - target_T) * (T1 - target_T) <= 0.0:
            if abs(T1 - T0) < 1e-20:
                alpha = 0.0
            else:
                alpha = (target_T - T0) / (T1 - T0)
            x_interp = xs[i] + alpha * (xs[i + 1] - xs[i])
            state_interp = ys[:, i] + alpha * (ys[:, i + 1] - ys[:, i])
            T_interp = final_temperature_from_state(state_interp, x_interp)
            return float(x_interp), state_interp, float(T_interp)

    idx = int(np.argmin(np.abs(temperatures - target_T)))
    return float(xs[idx]), ys[:, idx].copy(), float(temperatures[idx])


def write_summary_line(
    path: str,
    run_id: int,
    args: argparse.Namespace,
    ratios: Sequence[float],
    aT_ratio_cubed: float,
    neff: float,
) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            f"{run_id:d} "
            f"{args.llp_mass:.10e} {args.llp_lifetime:.10e} {args.nbins:d} {args.llp_abundance:.10e} "
            f"{args.llp_pion_branching:.10e} {args.llp_muon_branching:.10e} "
            f"{args.llp_two_nu_decay_e:.10e} {args.llp_two_nu_decay_mu:.10e} {args.llp_two_nu_decay_tau:.10e} "
            f"{ratios[0]:.10e} {ratios[1]:.10e} {ratios[2]:.10e} "
            f"{aT_ratio_cubed:.10e} {neff:.10e}\n"
        )


def append_distribution_snapshot(
    handle,
    T_snap: float,
    x_snap: float,
    state_snap: np.ndarray,
) -> None:
    n = Momentum_Grid.n
    p_vals = Momentum_Grid.gridVals * me / x_snap

    f_e, f_mu, f_tau = np.split(state_snap[:3 * n], 3)

    distr_e = f_e / np.pi**2
    distr_mu = f_mu / np.pi**2
    distr_tau = f_tau / np.pi**2

    for p_val, de, dm, dt in zip(p_vals, distr_e, distr_mu, distr_tau):
        handle.write(f"{T_snap:.10e} {p_val:.10e} {de:.10e} {dm:.10e} {dt:.10e}\n")
    handle.write("\n")


def initialize_distribution_file(
    distribution_path: str,
    run_id: int,
) -> None:
    ensure_parent_dir(distribution_path)
    with open(distribution_path, "w", encoding="utf-8") as handle:
        handle.write(f"# run_id = {run_id}\n")
        handle.write("# columns: T[MeV] p[MeV] f_nu_e(T,p) f_nu_mu(T,p) f_nu_tau(T,p)\n")
        handle.write("# p is the physical momentum used by the code: p = y * me / x = T * y / z\n")
        handle.write(
            "# Each flavor distribution is normalized so that "
            "integral f_nu_flavor(T,p) * p^2 dp = n_(nu_flavor + antinu_flavor)\n\n"
        )


def append_thermodynamics_snapshot(
    handle,
    T_snap: float,
    x_snap: float,
    state_snap: np.ndarray,
) -> None:
    n = Momentum_Grid.n
    t_snap = float(state_snap[3 * n + 1])
    rho_em = em_energy_density_from_state(state_snap, x_snap)
    rho_nu_e = neutrino_plus_antineutrino_energy_density_from_state(state_snap, 0, x_snap)
    rho_nu_mu = neutrino_plus_antineutrino_energy_density_from_state(state_snap, 1, x_snap)
    rho_nu_tau = neutrino_plus_antineutrino_energy_density_from_state(state_snap, 2, x_snap)
    n_nu_e = neutrino_plus_antineutrino_number_density_from_state(state_snap, 0, x_snap)
    n_nu_mu = neutrino_plus_antineutrino_number_density_from_state(state_snap, 1, x_snap)
    n_nu_tau = neutrino_plus_antineutrino_number_density_from_state(state_snap, 2, x_snap)
    a_snap = x_snap / me

    handle.write(
        f"{T_snap:.10e} {t_snap:.10e} {rho_em:.10e} "
        f"{rho_nu_e:.10e} {rho_nu_mu:.10e} {rho_nu_tau:.10e} "
        f"{n_nu_e:.10e} {n_nu_mu:.10e} {n_nu_tau:.10e} {a_snap:.10e}\n"
    )


def initialize_thermodynamics_file(
    thermodynamics_path: str,
    run_id: int,
) -> None:
    ensure_parent_dir(thermodynamics_path)
    with open(thermodynamics_path, "w", encoding="utf-8") as handle:
        handle.write(f"# run_id = {run_id}\n")
        handle.write(
            "# columns: T[MeV] t(T)[s] rho_EM(T)[MeV^4] rho_nu_e(T)[MeV^4] rho_nu_mu(T)[MeV^4] "
            "rho_nu_tau(T)[MeV^4] n_nu_e(T)[MeV^3] n_nu_mu(T)[MeV^3] n_nu_tau(T)[MeV^3] a(T)\n"
        )


def write_sampled_outputs(
    distribution_path: str,
    thermodynamics_path: str,
    targets: Sequence[float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    temperatures = ys[3 * Momentum_Grid.n, :] / xs * me

    with open(distribution_path, "a", encoding="utf-8") as distr_handle, open(
        thermodynamics_path, "a", encoding="utf-8"
    ) as thermo_handle:
        for target_T in targets:
            x_snap, state_snap, T_snap = interpolate_state_at_temperature(xs, ys, temperatures, target_T)
            append_distribution_snapshot(distr_handle, T_snap, x_snap, state_snap)
            append_thermodynamics_snapshot(thermo_handle, T_snap, x_snap, state_snap)


def make_stop_at_temperature_event(target_T: float):
    def stop_at_temperature(x, state, *_args):
        return final_temperature_from_state(state, x) - target_T

    stop_at_temperature.terminal = True
    stop_at_temperature.direction = -1.0
    return stop_at_temperature


def solve_until_temperature(
    rhs,
    x_start: float,
    x_end_initial: float,
    x_end_max: float,
    state_start: np.ndarray,
    rhs_args: Tuple,
    target_T: float,
    method: str,
    atol: float,
    rtol: float,
) -> SimpleNamespace:
    xs = []
    ys = []
    nfev_total = 0

    current_x = float(x_start)
    current_state = state_start.copy()
    current_x_end = float(x_end_initial)

    while True:
        event = make_stop_at_temperature_event(target_T)
        sol = solve_ivp(
            rhs,
            [current_x, current_x_end],
            current_state,
            args=rhs_args,
            method=method,
            t_eval=None,
            atol=atol,
            rtol=rtol,
            dense_output=False,
            events=event,
        )

        nfev_total += sol.nfev
        if xs:
            xs.append(sol.t[1:])
            ys.append(sol.y[:, 1:])
        else:
            xs.append(sol.t)
            ys.append(sol.y)

        if not sol.success:
            return SimpleNamespace(
                success=False,
                message=sol.message,
                t=np.concatenate(xs),
                y=np.concatenate(ys, axis=1),
                nfev=nfev_total,
                reached_Tfinal=False,
            )

        reached_Tfinal = len(sol.t_events) > 0 and len(sol.t_events[0]) > 0
        current_T = final_temperature_from_state(sol.y[:, -1], sol.t[-1])
        if reached_Tfinal or current_T <= target_T:
            return SimpleNamespace(
                success=True,
                message=sol.message,
                t=np.concatenate(xs),
                y=np.concatenate(ys, axis=1),
                nfev=nfev_total,
                reached_Tfinal=True,
            )

        if current_x_end >= x_end_max:
            return SimpleNamespace(
                success=True,
                message=f"Tfinal was not reached before max x = {x_end_max:g}.",
                t=np.concatenate(xs),
                y=np.concatenate(ys, axis=1),
                nfev=nfev_total,
                reached_Tfinal=False,
            )

        current_x = float(sol.t[-1])
        current_state = sol.y[:, -1].copy()
        current_x_end = min(float(x_end_max), max(2.0 * current_x_end, current_x + 1.0))
        print(
            f"Extending integration ceiling to x = {current_x_end:.6e} "
            f"to reach Tfinal = {target_T:.6e} MeV"
        )


def main() -> int:
    args = parse_args()

    output_dir = build_output_dir(args.output_folder)
    output_file = str(output_dir / args.name_output)
    use_decay_probabilities = USE_DECAY_PROBABILITIES_CFG and (not args.no_decay_probabilities)

    initialize_output_file(output_file)
    existing_ids = read_existing_ids(output_file)
    run_id = generate_unique_run_id(existing_ids)

    z_0 = np.array([args.z0], dtype=float)
    t_0 = np.array([args.t0], dtype=float)
    x_span = [float(args.x_start), float(args.x_end)]
    integration_x_end_max = float(args.max_x_end)

    injection_cutoff_time_s = args.llp_lifetime * args.lifetime_factor
    stop_point = build_stop_point(args.t0 + injection_cutoff_time_s, timeScaleFactorFile)
    llp_count = args.llp_abundance * (x_span[0] / me) ** 3
    grid_stop_x = estimate_grid_stop_x(
        target_elapsed_time_s=injection_cutoff_time_s,
        table_path=timeScaleFactorFile,
        x_start=x_span[0],
        x_end=integration_x_end_max,
        llp_count=llp_count,
        llp_lifetime=args.llp_lifetime,
        llp_mass=args.llp_mass,
        t_start=args.t0,
    )

    limits = [SMComovingLimit]

    if args.llp_muon_branching > 0 or args.llp_pion_branching > 0:
        limits.append(GRID_INJECTION_SAFETY * grid_stop_x * mmu / (2.0 * me))

    if args.llp_two_nu_decay_e + args.llp_two_nu_decay_mu + args.llp_two_nu_decay_tau > 0:
        limits.append(GRID_INJECTION_SAFETY * grid_stop_x * args.llp_mass / (2.0 * me))

    thermal_collision_limit = estimate_thermal_collision_grid_limit(
        grid_stop_x=grid_stop_x,
        llp_mass=args.llp_mass,
        llp_abundance=args.llp_abundance,
        llp_lifetime=args.llp_lifetime,
        T_start=args.Tstart,
    )
    if thermal_collision_limit > 0.0:
        limits.append(thermal_collision_limit)

    ylimit = max(limits)

    Momentum_Grid.setupGrid(float(ylimit), int(args.nbins))
    Distributions.initDistributions(args.llp_mass)

    system_nudecoupling_module.set_debugging_mode(args.ifDebugging)

    print(
        f"Run id={run_id} | y_max={ylimit:.4f} | n_grid={args.nbins} | "
        f"radiationStopPoint={stop_point:.6f} | cutoffTime={injection_cutoff_time_s:.6e}s | "
        f"gridStopX={grid_stop_x:.6f} | gridSafety={GRID_INJECTION_SAFETY:.2f} | "
        f"Tfinal={args.Tfinal:.6e} MeV | solver={args.solver} | "
        f"no_interactions={args.no_interactions} | ifDebugging={args.ifDebugging}"
    )

    decay_handler = make_decay_handler(use_decay_probabilities)

    f_nue_0 = fermi_dirac_zero_mu(Momentum_Grid.gridVals / z_0[0])
    f_numu_0 = fermi_dirac_zero_mu(Momentum_Grid.gridVals / z_0[0])
    f_nutau_0 = fermi_dirac_zero_mu(Momentum_Grid.gridVals / z_0[0])
    sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

    llp_branchings = [
        args.llp_muon_branching,
        args.llp_pion_branching,
        args.llp_two_nu_decay_e,
        args.llp_two_nu_decay_mu,
        args.llp_two_nu_decay_tau,
    ]
    arg_list = (
        llp_count,
        args.llp_lifetime,
        args.llp_mass,
        llp_branchings,
        stop_point,
        decay_handler,
        args.no_interactions,
        injection_cutoff_time_s,
        args.t0,
    )

    start = time.time()
    sol = solve_until_temperature(
        rhs=System_Nudec,
        x_start=x_span[0],
        x_end_initial=x_span[1],
        x_end_max=integration_x_end_max,
        state_start=sys_values_0,
        rhs_args=arg_list,
        target_T=args.Tfinal,
        method=args.solver,
        atol=args.atol,
        rtol=args.rtol,
    )
    runtime = time.time() - start

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    if not sol.reached_Tfinal:
        raise RuntimeError(
            f"{sol.message} Final T_gamma = "
            f"{final_temperature_from_state(sol.y[:, -1], sol.t[-1]):.10e} MeV. "
            "Increase --max-x-end."
        )

    final_state = sol.y[:, -1]
    x_fin = float(sol.t[-1])
    T_start = final_temperature_from_state(sys_values_0, x_span[0])
    T_fin = final_temperature_from_state(final_state, x_fin)
    t_fin = float(final_state[3 * Momentum_Grid.n + 1])

    neff = calc_neff(final_state)
    n_gamma_fin = photon_number_density(T_fin)
    ratios = [
        neutrino_plus_antineutrino_number_density_from_state(final_state, flavor_index=i, x=x_fin) / n_gamma_fin
        for i in range(3)
    ]
    aT_ratio_cubed = aT_ratio_cubed_from_states(
        state_start=sys_values_0,
        x_start=x_span[0],
        state_fin=final_state,
        x_fin=x_fin,
    )

    write_summary_line(output_file, run_id, args, ratios, aT_ratio_cubed, neff)

    output_dir_str = str(output_dir)
    distr_path = os.path.join(output_dir_str, f"{run_id}_nue-distr.txt")
    thermo_path = os.path.join(output_dir_str, f"{run_id}_thermodynamics.txt")

    initialize_distribution_file(distr_path, run_id)
    initialize_thermodynamics_file(thermo_path, run_id)

    targets = build_temperature_targets(T_start, T_fin)
    write_sampled_outputs(
        distribution_path=distr_path,
        thermodynamics_path=thermo_path,
        targets=targets,
        xs=sol.t,
        ys=sol.y,
    )

    print(f"Summary appended to: {output_file}")
    print(f"Output folder       = {output_dir_str}")
    print(f"Distribution snapshots written to: {distr_path}")
    print(f"Thermodynamics snapshots written to: {thermo_path}")
    print(f"T_start = {T_start:.10e} MeV")
    print(f"T_fin   = {T_fin:.10e} MeV")
    print(f"t_fin   = {t_fin:.10e} s")
    if t_fin - args.t0 < injection_cutoff_time_s:
        print(
            "WARNING: final time is before the LLP injection cutoff. "
            "N_eff is not a post-decay value; increase --max-x-end or lower --Tfinal."
        )
    print(f"n_nu_e/n_gamma   = {ratios[0]:.10e}")
    print(f"n_nu_mu/n_gamma  = {ratios[1]:.10e}")
    print(f"n_nu_tau/n_gamma = {ratios[2]:.10e}")
    print(f"((a*T)_start/(a*T)_fin)^3 = {aT_ratio_cubed:.10e}")
    print(f"N_eff            = {neff:.10e}")
    print(f"Accepted steps    = {len(sol.t) - 1}")
    print(f"RHS evaluations   = {sol.nfev}")
    print(f"Runtime           = {runtime:.4f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
