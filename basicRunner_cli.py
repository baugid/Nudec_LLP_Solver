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
from typing import List, Sequence, Tuple

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
from globalParameters import (
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

SCRIPT_DIR = Path(__file__).resolve().parent


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
            f"(default chosen to reproduce globalParameters.py initialX={initialX})."
        ),
    )
    parser.add_argument(
        "--x-end",
        type=float,
        default=finalX,
        help=f"Final x value (default from globalParameters.py: {finalX})",
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
            "and subtract its energy moment from dz/dx only. Default comes from globalParameters.py. "
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
    if os.path.isabs(args.name_output):
        parser.error("--name-output must be a filename, not an absolute path")
    if not args.name_output.strip():
        parser.error("--name-output must not be empty")

    args.x_start = convert_Tstart_to_xstart(args.Tstart)
    return args


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_existing_ids(path: str) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids

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
    if os.path.exists(path):
        return

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# id llp_mass llp_lifetime nbins llp_abundance llp_pionBranching "
            "llp_muonBranching llp_twoNuDecayE llp_twoNuDecayMu llp_twoNuDecayTau "
            "n_nu_e/n_gamma n_nu_mu/n_gamma n_nu_tau/n_gamma "
            "aT_ratio_cubed Neff\n"
        )


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


def calc_neff(state: np.ndarray) -> float:
    n = Momentum_Grid.n

    rho1 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[:n])
    rho2 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[n:2 * n])
    rho3 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals**3 * state[2 * n:3 * n])
    rho4 = np.sum(
        Momentum_Grid.gridWeights
        * Momentum_Grid.gridVals**3
        * 1.0
        / (np.exp(Momentum_Grid.gridVals / 1.0) + 1.0)
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
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
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

    stop_point = build_stop_point(args.llp_lifetime * args.lifetime_factor, timeScaleFactorFile)

    limits = [SMComovingLimit]

    if args.llp_muon_branching > 0 or args.llp_pion_branching > 0:
        limits.append(stop_point * mmu / (2.0 * me))

    if args.llp_two_nu_decay_e + args.llp_two_nu_decay_mu + args.llp_two_nu_decay_tau > 0:
        limits.append(stop_point * args.llp_mass / (2.0 * me))

    ylimit = max(limits)

    Momentum_Grid.setupGrid(float(ylimit), int(args.nbins))
    Distributions.initDistributions(args.llp_mass)

    system_nudecoupling_module.set_debugging_mode(args.ifDebugging)

    print(
        f"Run id={run_id} | y_max={ylimit:.4f} | n_grid={args.nbins} | "
        f"stopPoint={stop_point:.6f} | solver={args.solver} | "
        f"no_interactions={args.no_interactions} | ifDebugging={args.ifDebugging}"
    )

    decay_handler = make_decay_handler(use_decay_probabilities)

    f_nue_0 = 1.0 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1.0)
    f_numu_0 = 1.0 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1.0)
    f_nutau_0 = 1.0 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1.0)
    sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

    llp_branchings = [
        args.llp_muon_branching,
        args.llp_pion_branching,
        args.llp_two_nu_decay_e,
        args.llp_two_nu_decay_mu,
        args.llp_two_nu_decay_tau,
    ]
    arg_list = (
        args.llp_abundance * (x_span[0] / me) ** 3,
        args.llp_lifetime,
        args.llp_mass,
        llp_branchings,
        stop_point,
        decay_handler,
        args.no_interactions,
    )

    start = time.time()
    sol = solve_ivp(
        System_Nudec,
        x_span,
        sys_values_0,
        args=arg_list,
        method=args.solver,
        t_eval=None,
        atol=args.atol,
        rtol=args.rtol,
        dense_output=False,
    )
    runtime = time.time() - start

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    final_state = sol.y[:, -1]
    x_fin = float(sol.t[-1])
    T_start = final_temperature_from_state(sys_values_0, x_span[0])
    T_fin = final_temperature_from_state(final_state, x_fin)

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
