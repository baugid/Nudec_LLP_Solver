import time
import numpy as np
from scipy import integrate

# Physics modules
try:
    from Thermodynamics.Thermodynamics_ideal_gas import *
except ModuleNotFoundError:
    from Thermodynamics_ideal_gas import *

try:
    from Thermodynamics.Thermal_QED_corrections import *
except ModuleNotFoundError:
    from Thermal_QED_corrections import *

try:
    from Collision_term.Collision_term_diagonal import *
    from Collision_term.Collision_term_diagonal import Collision_term_self_only
except ModuleNotFoundError:
    from Collision_term_diagonal import *
    from Collision_term_diagonal import Collision_term_self_only

# Constants twice: symbols, and module alias for runtime flag
from Constants import *
import Constants as C

import Momentum_Grid
from Averaged_Nu_Osc import *
import Distributions
from globalParameters import debugOutput, ifDebugging


def rotationMatrix(th, d):
    sth = np.sin(th)
    cth = np.cos(th)
    zero = np.zeros(np.shape(sth))
    ones = np.ones(np.shape(sth))
    if d == 1:
        ret = np.array([[ones, zero, zero],
                        [zero, cth, sth],
                        [zero, -sth, cth]])
    elif d == 2:
        ret = np.array([[cth, zero, sth],
                        [zero, ones, zero],
                        [-sth, zero, cth]])
    else:
        ret = np.array([[cth, sth, zero],
                        [-sth, cth, zero],
                        [zero, zero, ones]])
    shape = np.shape(ret)
    if len(shape) > 2:
        ret = np.moveaxis(ret, range(2, len(shape)), range(0, len(shape)-2))
    return ret


def calcTransfers(T, momenta):
    """
    Return the 3x3 flavor transfer probability matrix P_αβ(p) per momentum bin.
    If oscillations are disabled, return identity for each momentum.
    """
    if not getattr(C, "ifOsc", True):
        I = np.eye(3, dtype=float)
        return np.broadcast_to(I, (np.size(momenta), 3, 3)).copy()

    A_MSW = MSWPrefactor * momenta**2 * T**4

    ang12 = np.arctan2(ds12, dc12 + A_MSW / Dm21sq) / 2
    ang23 = np.arctan2(ds23, dc23 + A_MSW / Dm31sq) / 2
    ang13 = np.arctan2(ds13, dc13 + A_MSW / Dm31sq) / 2

    pmns = rotationMatrix(ang23, 1) @ rotationMatrix(ang13, 2) @ rotationMatrix(ang12, 3)
    pmns_sq = pmns ** 2  # real case

    # P = |U|^2 |U|^2^T per momentum
    P = pmns_sq @ np.swapaxes(pmns_sq, -1, -2)

    # Normalize rows to guard against small numerical drift
    P = np.maximum(P, 0.0)
    row_sum = np.sum(P, axis=-1, keepdims=True)
    P = P / np.maximum(row_sum, 1e-16)
    return P


callNumber = 0

# Runtime debugging switch. When enabled, the code computes the pure nu-self
# collision term a second time, forms its contribution to the neutrino energy
# moment, and subtracts that moment ONLY inside dzdx. The neutrino spectra
# evolution is unchanged. This is intended strictly as a debugging cross-check.
DEBUG_SUBTRACT_SELF_FROM_Z = bool(ifDebugging)
DEBUG_PRINT_SELF_Z_TEST = bool(ifDebugging)
DEBUG_SELF_Z_LOG_EVERY = 50


def set_debugging_mode(flag: bool):
    """Enable/disable the nu-self debugging cross-check at runtime."""
    global DEBUG_SUBTRACT_SELF_FROM_Z, DEBUG_PRINT_SELF_Z_TEST
    enabled = bool(flag)
    DEBUG_SUBTRACT_SELF_FROM_Z = enabled
    DEBUG_PRINT_SELF_Z_TEST = enabled


def get_debugging_mode() -> bool:
    """Return True when the extra nu-self debugging path is enabled."""
    return bool(DEBUG_SUBTRACT_SELF_FROM_Z or DEBUG_PRINT_SELF_Z_TEST)


def System_Nudec(x, sys_values, llp_count, llp_lifetime, llp_mass, branching_fractions, stopPoint, decayHandler, no_interactions=False):
    """
    Right-hand side of the coupled system in variable x.
    """
    global callNumber
    callNumber += 1

    if debugOutput:
        print(callNumber)
        print(x)

    if callNumber == 1 or callNumber % 5 == 0:
        print(f"Progress: x = {x:.5f}")

    start = time.time()

    # Unpack state
    n = Momentum_Grid.n
    y = Momentum_Grid.gridVals
    w = Momentum_Grid.gridWeights

    f_nue, f_numu, f_nutau = np.split(sys_values[:3 * n], 3)
    f_nue_bar = f_nue.copy()
    f_numu_bar = f_numu.copy()
    f_nutau_bar = f_nutau.copy()

    z = sys_values[3 * n]
    t = sys_values[3 * n + 1]

    # Ideal-gas energy densities and auxiliaries
    rho_e_bar, rho_nu_bar = Energy_density_ideal_gas(
        x, z, f_nue, f_numu, f_nutau,
        f_nue_bar, f_numu_bar, f_nutau_bar
    )
    J, Y = Functions_in_z_ideal_gas(x, z)

    # Thermal QED corrections
    delta_me = Thermal_QED_corrections_to_me(x, z)
    rho_2, rho_3 = Thermal_QED_corrections_to_energy_density(x, z)
    G2_1, G2_2, G3_1, G3_2 = Thermal_QED_corrections_to_z(x, z)

    # LLP density (comoving)
    llp_density = llp_count * np.exp(-t / llp_lifetime) * (me / x) ** 3

    # Total energy density in comoving volume
    rho_bar = (
        (np.pi ** 2 * z ** 4) / 15
        + rho_nu_bar
        + rho_e_bar
        + rho_2
        + rho_3
        + llp_mass * llp_density * (x / me) ** 4
    )

    # Hubble (comoving and physical)
    Hubble = 1 / mpl * ((8 * np.pi) / 3 * rho_bar) ** 0.5
    trueHubble = Hubble * (me / x) ** 2

    # LLP decay term
    dllpdt = llp_density / (llp_lifetime / hbar)

    # Injection (from LLPs if any)
    df_nudx = np.zeros((3, n))
    momentumVals = y * me / x

    if x < stopPoint:
        decayProbabilites = decayHandler(z / x * me)

        for branching, distri in zip(branching_fractions, Distributions.getDistribution):
            # Important: skip channels with zero branching entirely.
            # Otherwise distri(...) is still evaluated and may fail
            # by demanding momenta outside the active grid even though
            # that channel contributes exactly zero.
            if branching == 0.0:
                continue

            df_nudx += (
                (1 / (2 * trueHubble * x))
                * branching
                * dllpdt
                * distri(momentumVals, x, decayProbabilites)
                * (2 * np.pi ** 2) / (momentumVals ** 2)
            )  # factor 1/2: ν and ν̄

    # Oscillation-averaged flavor transfer on injected piece
    transferProps = calcTransfers(z / x * me, momentumVals)
    df_nudx = np.einsum("ijk,ki->ji", transferProps, df_nudx)

    # Collision terms + oscillation mixing for the thermal piece
    df_nuedx = df_nudx[0]
    df_numudx = df_nudx[1]
    df_nutaudx = df_nudx[2]

    delta_self_x = 0.0
    need_self_diag = (DEBUG_SUBTRACT_SELF_FROM_Z or DEBUG_PRINT_SELF_Z_TEST)

    if not no_interactions:
        for ni in range(n):
            i = y[ni]
            pref = Hubble ** (-1) * (me ** 3 * x ** (-4))

            Coll_diag = Collision_term_diagonal(
                x, z, ni, i,
                f_nue, f_numu, f_nutau,
                f_nue_bar, f_numu_bar, f_nutau_bar,
                delta_me
            )

            # Apply transfer matrix row for this momentum
            Coll_nue, Coll_numu, Coll_nutau = transferProps[ni] @ Coll_diag
            df_nuedx[ni] += pref * Coll_nue
            df_numudx[ni] += pref * Coll_numu
            df_nutaudx[ni] += pref * Coll_nutau

            if need_self_diag:
                Self_diag = Collision_term_self_only(
                    x, z, ni, i,
                    f_nue, f_numu, f_nutau,
                    f_nue_bar, f_numu_bar, f_nutau_bar,
                    delta_me
                )
                Self_nue, Self_numu, Self_nutau = transferProps[ni] @ Self_diag
                delta_self_x += w[ni] * y[ni] ** 3 * pref * (Self_nue + Self_numu + Self_nutau)

    # Photon temperature evolution
    dzdx = np.zeros(1)

    drho_nudx = (1 / (2 * np.pi ** 2 * z ** 3)) * np.sum(
        y ** 3 * (df_nuedx + df_numudx + df_nutaudx) * w
    )

    if need_self_diag:
        delta_self_x *= 1 / (2 * np.pi ** 2 * z ** 3)
    else:
        delta_self_x = 0.0

    if DEBUG_SUBTRACT_SELF_FROM_Z:
        drho_nudx_for_z = drho_nudx - delta_self_x
    else:
        drho_nudx_for_z = drho_nudx

    if x < stopPoint:
        drho_llpdx = -llp_mass * (x / me / z) ** 3 * (dllpdt / (trueHubble * x)) * x / me / 2
    else:
        drho_llpdx = 0.0

    dzdx[0] = (
        x / z * J - (drho_nudx_for_z + drho_llpdx) + G2_1 + G3_1
    ) / (
        x ** 2 / z ** 2 * J + Y + 2 * np.pi ** 2 / 15 + G2_2 + G3_2
    )

    if DEBUG_PRINT_SELF_Z_TEST and (callNumber == 1 or callNumber % DEBUG_SELF_Z_LOG_EVERY == 0):
        print(
            f"Self-z test: x = {x:.6e}, z = {z:.6e}, "
            f"drho_nudx = {drho_nudx:.6e}, "
            f"Delta_self = {delta_self_x:.6e}, "
            f"drho_nudx_for_z = {drho_nudx_for_z:.6e}, "
            f"subtracting = {DEBUG_SUBTRACT_SELF_FROM_Z}"
        )

    # Time equation
    dtdx = np.zeros(1)
    dtdx[0] = hbar / (trueHubble * x)

    diff_sys = np.concatenate((df_nuedx, df_numudx, df_nutaudx, dzdx, dtdx))

    if debugOutput:
        print("runtime:", time.time() - start)

    return diff_sys