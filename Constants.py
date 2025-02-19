"""
This file contains all physical constants used across the code
"""
import numpy as np

alpha = 7.2973525 * 10 ** (-3)  # Fine structure constant
e = (4 * np.pi * alpha) ** (1 / 2)  # electromagnetic couping
me = 0.5109989  # electron mass (MeV)
mmu = 105.7  # muon mass (MeV)
mpi = 135  # pion mass (MeV)

sW = 0.231 ** (1 / 2)  # Weak mixing angle
gL = 0.731  # 1/2 + sW**2
gLtilde = -0.269  # -1/2 + sW**2
gR = 0.231  # sW**2
GF = 1.1663787 * 10 ** (-11)  # Fermi coupling constant (MeV**(-2))
mW = 80.379 * 10 ** 3  # W boson mass (MeV)
mZ = mW * (1 - sW ** 2) ** (1 / 2)  # MeV #W boson mass
G = 6.70833 * 10 ** (-45)  # Gravitational constant (MeV**(-2))
mpl = (1 / G) ** (1 / 2)  # Planck mass (MeV)
hbar = 6.582e-22  # hbar (MeV*s)

# Neutrino mass and mixing NuFit-5.2
# The best-fit value in Normal Ordering from NuFit-5.2 http://www.nu-fit.org/?q=node/12

Dm21sq = 7.41 * 10 ** (-17)  # m2**2-m**1 (MeV**2)
Dm31sq = 2.511 * 10 ** (-15)  # m3**2-m1**2 (MeV**2)
s12 = 0.303 ** (1 / 2)  # sin theta_12
s23 = 0.572 ** (1 / 2)  # sin theta_23
s13 = 0.02203 ** (1 / 2)  # sin theta_13
c12 = (1 - s12 ** 2) ** (1 / 2)  # cos theta_12
c23 = (1 - s23 ** 2) ** (1 / 2)  # cos theta_23
c13 = (1 - s13 ** 2) ** (1 / 2)  # cos theta_13
deltaCP = 0  # We assume it for simplicity in this version

# trig functions above at double the angle
ds12 = np.sin(2 * np.arcsin(s12))
ds23 = np.sin(2 * np.arcsin(s23))
ds13 = np.sin(2 * np.arcsin(s13))
dc12 = np.cos(2 * np.arccos(c12))
dc23 = np.cos(2 * np.arccos(c23))
dc13 = np.cos(2 * np.arccos(c13))

MSWPrefactor = 8.65785 * GF / mW ** 2

# PMNS matrix (with deltaCP = 0)

Ue1 = c12 * c13
Ue2 = s12 * c13
Ue3 = s13
Umu1 = -s12 * c23 - c12 * s23 * s13
Umu2 = c12 * c23 - s12 * s23 * s13
Umu3 = s23 * c13
Utau1 = s12 * s23 - c12 * c23 * s13
Utau2 = -c12 * s23 - s12 * c23 * s13
Utau3 = c23 * c13
