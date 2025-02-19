"""
This file contains the central method, which implements the system of differential equations.
Furthermore, some helper methods are added.
"""

import time
import numpy as np

from Thermodynamics.Thermodynamics_ideal_gas import *
from Thermodynamics.Thermal_QED_corrections import *
from Collision_term.Collision_term_diagonal import *
from Constants import *
import Momentum_Grid
import Distributions
from GlobalParameters import *


def rotationMatrix(th, d):
    """
    Calculate the rotation matrix for a given rotation angle and dimension.
    Note that the code is compatible with np arrays of angles for th.
    :param th: (list of) Rotation angle in radians.
    :param d: Index of the dimension, which remain constant.
    :return: A (list of) 3x3 rotation matrix.
    """
    sth = np.sin(th)
    cth = np.cos(th)
    zero = np.zeros(np.shape(sth))
    ones = np.ones(np.shape(sth))

    if d == 1:
        ret = np.array([[ones, zero, zero], [zero, cth, sth], [zero, -sth, cth]])
    elif d == 2:
        ret = np.array([[cth, zero, sth], [zero, ones, zero], [-sth, zero, cth]])
    else:
        ret = np.array([[cth, sth, zero], [-sth, cth, zero], [zero, zero, ones]])

    # ret is created as a matrix of vectors, this is rectified here
    shape = np.shape(ret)
    if len(shape) > 2:
        ret = np.moveaxis(ret, range(2, len(shape)), range(0, len(shape) - 2))
    return ret


def calcMixingMatrixes(T, momenta):
    """
    Calculate the averaged neutrino mixing matrices for a given temperature and momenta.
    :param T: The current temperature in MeV.
    :param momenta: A list of momenta values.
    :return: A list of mixing matrices with the same dimensions as momenta.
    """
    A_MSW = MSWPrefactor * momenta ** 2 * T ** 4

    # Calculate the angles. Note the arctan2 to ensure correct normalization
    ang12 = np.arctan2(ds12, dc12 + A_MSW / Dm21sq) / 2
    ang23 = np.ones(np.shape(momenta)) * np.arcsin(s23)
    ang13 = np.arctan2(ds13, dc13 + A_MSW / Dm31sq) / 2

    # From this the matrix of absolute values of entries in the PMNS matrix is calculated
    pmns = rotationMatrix(ang23, 1) @ rotationMatrix(ang13, 2) @ rotationMatrix(ang12, 3)
    pmns **= 2

    # The squared absolute values times the same matrix transposed gives the averaged probability
    mixingProbability = pmns @ np.swapaxes(pmns, -1, -2)
    return mixingProbability

# globals for progress indication
callNumber = 0
lastPrintTime = 0


def System_Nudec(x, sys_values, llp_count, llp_lifetime, llp_mass, branching_fractions, stopPoint, decayHandler):
    global callNumber, lastPrintTime
    callNumber += 1

    # Unpack sys_values
    f_nue, f_numu, f_nutau = np.split(sys_values[:3 * Momentum_Grid.n], 3)

    f_nue_bar = f_nue.copy()
    f_numu_bar = f_numu.copy()
    f_nutau_bar = f_nutau.copy()

    z = sys_values[3 * Momentum_Grid.n]
    t = sys_values[3 * Momentum_Grid.n + 1]

    if informationDuringRun and time.time() - lastPrintTime > printRate:
        lastPrintTime = time.time()
        print(f"[{time.time():.2f}] Starting step {callNumber} for t = {t:.3}s.")

    # e^\pm energy density, e^\pm pressure, neutrino and anti-neutrino energy density in comoving volume with ideal gas limit
    rho_e_bar, rho_nu_bar = Energy_density_ideal_gas(x, z, f_nue, f_numu, f_nutau, f_nue_bar, f_numu_bar, f_nutau_bar)
    J, Y = Functions_in_z_ideal_gas(x, z)

    # Thermal QED corrections to electron mass, energy density and pressure
    delta_me = Thermal_QED_corrections_to_me(x, z)
    rho_2, rho_3 = Thermal_QED_corrections_to_energy_density(x, z)
    G2_1, G2_2, G3_1, G3_2 = Thermal_QED_corrections_to_z(x, z)

    # Current LLP density
    llp_density = llp_count * np.exp(-t / llp_lifetime) * (me / x) ** 3

    # Total energy density in comoving volume (i.e., \bar{\rho} in (A.2) in 2210.10307)
    rho_bar = (np.pi ** 2 * z ** 4) / 15 + rho_nu_bar + rho_e_bar + rho_2 + rho_3 + llp_mass * llp_density * (
            x / me) ** 4

    # Hubble rates
    comovingHubble = 1 / mpl * ((8 * np.pi) / 3 * rho_bar) ** (1 / 2)
    trueHubble = comovingHubble * (me / x) ** 2

    # Compute LLP contributions
    dllpdt = llp_density / (llp_lifetime / hbar)

    df_nudx = np.zeros((3, Momentum_Grid.n))
    momentumVals = Momentum_Grid.gridVals * me / x

    if x < stopPoint:
        decayProbabilites = decayHandler(t)
        for branching, distri in zip(branching_fractions, Distributions.getDistribution):
            df_nudx += 1 / (2 * trueHubble * x) * branching * dllpdt * distri(momentumVals, x, decayProbabilites) * (
                    2 * np.pi ** 2) / (momentumVals ** 2)  # factor 1/2 is needed since we produce nu and nubar

    # neutrino oscillations in injected neutrinos
    transferProbability = calcMixingMatrixes(z / x * me, momentumVals)
    df_nudx = np.einsum("ijk,ki->ji", transferProbability, df_nudx)

    # Boltzmann equations for the neutrino distribution function
    df_nuedx, df_numudx, df_nutaudx = df_nudx

    # Calculate the collision term for each bin
    for ni in range(Momentum_Grid.n):

        i = Momentum_Grid.gridVals[ni]

        # Compute the collision terms
        Coll_diag = Collision_term_diagonal(x, z, ni, i, f_nue, f_numu, f_nutau, f_nue_bar, f_numu_bar, f_nutau_bar,
                                            delta_me)

        # Mix the neutrinos
        Coll_nue, Coll_numu, Coll_nutau = transferProbability[ni] @ Coll_diag

        df_nuedx[ni] += comovingHubble ** (-1) * (me ** 3 * x ** (-4) * Coll_nue)
        df_numudx[ni] += comovingHubble ** (-1) * (me ** 3 * x ** (-4) * Coll_numu)
        df_nutaudx[ni] += comovingHubble ** (-1) * (me ** 3 * x ** (-4) * Coll_nutau)

    # Photon temperature evolution (Continuity equation)
    dzdx = np.zeros(1)

    y = Momentum_Grid.gridVals

    drho_nudx = 1 / (2 * np.pi ** 2 * z ** 3) * np.sum(
        y ** 3 * (df_nuedx + df_numudx + df_nutaudx) * Momentum_Grid.gridWeights)

    # Total change of LLP energy density
    if x < stopPoint:
        drho_llpdx = -llp_mass * (x / me / z) ** 3 * (dllpdt / (trueHubble * x)) * x / me / 2
    else:
        drho_llpdx = 0

    dzdx[0] = (x / z * J - (drho_nudx + drho_llpdx) + G2_1 + G3_1) / (
            x ** 2 / z ** 2 * J + Y + 2 * np.pi ** 2 / 15 + G2_2 + G3_2)  # (A.15)

    # time differential equation from Hubble definition:
    dtdx = np.zeros(1)
    dtdx[0] = hbar / (trueHubble * x)

    diff_sys = np.concatenate((df_nuedx, df_numudx, df_nutaudx, dzdx, dtdx))

    return diff_sys
