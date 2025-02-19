"""
Here the thermal QED corrections to energy density, pressure and electron mass are implemented.
See Section 2.4 and Appendix A in K. Akita and M. Yamaguchi, arXiv: 2210.10307 for details. The equation numbers always reference 2210.10307.
"""
import numpy as np
from scipy import integrate
from Constants import *
import Momentum_Grid


# Thermal QED corrections to Hubble parameter (i.e., energy density)
def Thermal_QED_corrections_to_energy_density(x, z):
    rho_2 = -P_2(x, z) + z * dP_2dz(x, z)

    rho_3 = e ** 3 * x ** 2 / (8 * np.pi ** 4) * I(x, z) ** (1 / 2) * dIdz(x, z)

    return rho_2, rho_3


def P_2(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)

    Integrand_tmp = y ** 2 / (y ** 2 + x ** 2) ** (1 / 2) * 2 / (np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1)

    tmp = integrate.simpson(Integrand_tmp, x=y)

    P_2 = -e ** 2 * z ** 2 / (12 * np.pi ** 2) * tmp - e ** 2 / (8 * np.pi ** 4) * tmp ** 2

    return P_2


def dP_2dz(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)

    Integrand_tmp1 = y ** 2 / (y ** 2 + x ** 2) ** (1 / 2) * 2 / (np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1)

    Integrand_tmp2 = 2 * y ** 2 / z ** 2 * np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) / (
            np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1) ** 2

    tmp1 = integrate.simpson(Integrand_tmp1, x=y)

    tmp2 = integrate.simpson(Integrand_tmp2, x=y)

    dP_2dz = -e ** 2 * z / (6 * np.pi ** 2) * tmp1 - e ** 2 * z ** 2 / (12 * np.pi ** 2) * tmp2 - e ** 2 / (
            4 * np.pi ** 2) * tmp1 * tmp2

    return dP_2dz


def I(x, z):  # (2.54)
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)

    Integrand_I = (2 * y ** 2 + x ** 2) / (y ** 2 + x ** 2) ** (1 / 2) * 2 / (
            np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1)

    I = integrate.simpson(Integrand_I, x=y)

    return I


def dIdz(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)

    Integrand_dIdz = 2 * (2 * y ** 2 + x ** 2) / z ** 2 * np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) / (
            np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1) ** 2

    dIdz = integrate.simpson(Integrand_dIdz, x=y)

    return dIdz


def Thermal_QED_corrections_to_me(x, z):  # (A.6)
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)

    Integrand_me = y ** 2 / (y ** 2 + x ** 2) ** (1 / 2) * 1 / (np.exp((y ** 2 + x ** 2) ** (1 / 2) / z) + 1)

    delta_me = e ** 2 * z ** 2 / 6 + e ** 2 / (np.pi ** 2) * integrate.simpson(Integrand_me, x=y)

    return delta_me


def Thermal_QED_corrections_to_z(x, z):  # (A.16) and (A.17)
    w = x / z

    G2_1 = 2 * np.pi * alpha * ((1 / w) * (K(x, z) / 3 + 2 * K(x, z) ** 2 - J(x, z) / 6 - K(x, z) * J(x, z)) + (
            dKdw(x, z) / 6 - K(x, z) * dKdw(x, z) + dJdw(x, z) / 6 + dJdw(x, z) * K(x, z) + J(x, z) * dKdw(x, z)))

    G2_2 = -8 * np.pi * alpha * (
            K(x, z) / 6 + J(x, z) / 6 - K(x, z) ** 2 / 2 + K(x, z) * J(x, z)) + 2 * np.pi * alpha * w * (
                   dKdw(x, z) / 6 - K(x, z) * dKdw(x, z) + dJdw(x, z) / 6 + dJdw(x, z) * K(x, z) + J(x, z) * dKdw(x,
                                                                                                                  z))

    G3_1 = e ** 3 / (4 * np.pi) * (K(x, z) + w ** 2 / 2 * k(x, z)) ** (1 / 2) * (
            1 / w * (2 * J(x, z) - 4 * K(x, z)) - 2 * dJdw(x, z) - w ** 2 * djdw(x, z) - w * (
            2 * k(x, z) + j(x, z)) - (2 * J(x, z) + w ** 2 * j(x, z)) * (
                    w * (k(x, z) - j(x, z)) + dKdw(x, z)) / (2 * (2 * K(x, z) + w ** 2 * k(x, z))))

    G3_2 = e ** 3 / (4 * np.pi) * (K(x, z) + w ** 2 / 2 * k(x, z)) ** (1 / 2) * (
            (2 * J(x, z) + w ** 2 * j(x, z)) ** 2 / (2 * (2 * K(x, z) + w ** 2 * k(x, z))) - 2 / w * dYdw(x,
                                                                                                          z) - w * (
                    3 * dJdw(x, z) + w ** 2 * djdw(x, z)))

    return G2_1, G2_2, G3_1, G3_2


def K(x, z):  # (A.17)
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_K = u ** 2 / (u ** 2 + w ** 2) ** (1 / 2) * 1 / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1)

    K = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_K, x=y)

    return K


def J(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_J = u ** 2 * np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2

    J = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_J, x=y)

    return J


def Y(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_Y = u ** 4 * np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2

    Y = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_Y, x=y)

    return Y


def k(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_k = 1 / (u ** 2 + w ** 2) ** (1 / 2) * 1 / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1)

    k = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_k, x=y)

    return k


def j(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_j = np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2

    j = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_j, x=y)

    return j


def dKdw(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_dKdw = -w * u ** 2 * (
            1 / (u ** 2 + w ** 2) ** (3 / 2) * 1 / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) + 1 / (
            u ** 2 + w ** 2) * np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (
                    np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2)

    dKdw = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_dKdw, x=y)

    return dKdw


def dJdw(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_dJdw = w * u ** 2 / (u ** 2 + w ** 2) ** (1 / 2) * (
            np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2 - 2 * np.exp(
        2 * (u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 3)

    dJdw = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_dJdw, x=y)

    return dJdw


def dYdw(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_dYdw = w * u ** 4 / (u ** 2 + w ** 2) ** (1 / 2) * (
            np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2 - 2 * np.exp(
        2 * (u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 3)

    dYdw = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_dYdw, x=y)

    return dYdw


def djdw(x, z):
    y = np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)
    u = y / z
    w = x / z
    dudy = 1 / z

    Integrand_djdw = w / (u ** 2 + w ** 2) ** (1 / 2) * (
            np.exp((u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 2 - 2 * np.exp(
        2 * (u ** 2 + w ** 2) ** (1 / 2)) / (np.exp((u ** 2 + w ** 2) ** (1 / 2)) + 1) ** 3)

    djdw = dudy * 1 / (np.pi ** 2) * integrate.simpson(Integrand_djdw, x=y)

    return djdw
