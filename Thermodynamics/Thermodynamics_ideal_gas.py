"""
Here some thermodynamic quantities in the ideal gas limit are calculated. 
See Section 2.3 and Appendix A in K. Akita and M. Yamaguchi, arXiv: 2210.10307 for details. The equation numbers always reference 2210.10307.
"""
import numpy as np
from scipy import integrate
from Constants import *
import Momentum_Grid


def Energy_density_ideal_gas(x, z, f_nue, f_numu, f_nutau, f_nue_bar, f_numu_bar, f_nutau_bar):
    y = Momentum_Grid.gridVals

    Integrand_rho_e_bar = y ** 2 * (y ** 2 + x ** 2) ** (1 / 2) / (np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1)
    Integrand_rho_nu_bar = y ** 3 * (f_nue + f_numu + f_nutau + f_nue_bar + f_numu_bar + f_nutau_bar)

    # Energy density for e^\pm in comoving volume
    rho_e_bar = 2 / (np.pi ** 2) * np.sum(Integrand_rho_e_bar * Momentum_Grid.gridWeights)
    # Total neutrino and anti-neutrino energy density
    rho_nu_bar = 1 / (2 * np.pi ** 2) * np.sum(Integrand_rho_nu_bar * Momentum_Grid.gridWeights)

    return rho_e_bar, rho_nu_bar


def Functions_in_z_ideal_gas(x, z):
    J_return = J(x, z)
    Y_return = Y(x, z)

    return J_return, Y_return


def Functions_in_z_ideal_gas_mu_pi(x, z):
    Jmu_return = Jmu(x, z)

    Ymu_return = Ymu(x, z)

    Jpi_return = Jpi(x, z)

    Ypi_return = Ypi(x, z)

    return Jmu_return, Ymu_return, Jpi_return, Ypi_return


def J(x, z): #(A.17)
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_J = u ** 2 * np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2

    J = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_J, x=y)

    return J


def Y(x, z): #(A.17)
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_Y = u ** 4 * np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2

    Y = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_Y, x=y)

    return Y


def Jmu(x, z): #(A.17) for muons
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudi = 1 / z

    Integrand_Jmu = dudi * 1 / (np.pi ** 2) * u ** 2 * np.exp((u ** 2 + (mmu / me) ** 2 * w ** 2) ** (1 / 2)) / (
            np.exp((u ** 2 + (mmu / me) ** 2 * w ** 2) ** (1 / 2)) + 1) ** 2

    Jmu = integrate.simpson(Integrand_Jmu, x=y)

    return Jmu


def Jpi(x, z): #(A.17) for pions
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudi = 1 / z

    Integrand_Jpi = dudi * 1 / (np.pi ** 2) * u ** 2 * np.exp((u ** 2 + (mpi / me) ** 2 * w ** 2) ** (1 / 2)) / (
            np.exp((u ** 2 + (mpi / me) ** 2 * w ** 2) ** (1 / 2)) - 1) ** 2

    Jpi = integrate.simpson(Integrand_Jpi, x=y)

    return Jpi


def Ymu(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudi = 1 / z

    Integrand_Ymu = dudi * 1 / (np.pi ** 2) * u ** 4 * np.exp((u ** 2 + (mmu / me) ** 2 * w ** 2) ** (1 / 2)) / (
            np.exp((u ** 2 + (mmu / me) ** 2 * w ** 2) ** (1 / 2)) + 1) ** 2

    Ymu = integrate.simpson(Integrand_Ymu, x=y)

    return Ymu


def Ypi(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudi = 1 / z

    Integrand_Ypi = dudi * 1 / (np.pi ** 2) * u ** 4 * np.exp((u ** 2 + (mpi / me) ** 2 * w ** 2) ** (1 / 2)) / (
            np.exp((u ** 2 + (mpi / me) ** 2 * w ** 2) ** (1 / 2)) - 1) ** 2

    Ypi = integrate.simpson(Integrand_Ypi, x=y)

    return Ypi
