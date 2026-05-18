import numpy as np

# ========= Fundamental constants =========
alpha = 7.2973525e-3                  # fine-structure
e = (4 * np.pi * alpha) ** 0.5
me = 0.5109989                        # MeV
mmu = 105.7                           # MeV
mpi = 135.0                           # MeV

# sW = (0.231) ** 0.5
sW=(0.238)**0.5
#gL = 0.731
gL=0.727
#gLtilde = -0.269
gLtilde=-0.273
#gR = 0.231
gR=0.233
GF = 1.1663787e-11                    # MeV^-2
mW = 80.379e3                         # MeV
mZ = mW * (1 - sW**2) ** 0.5
G = 6.70833e-45                       # MeV^-2
mpl = (1.0 / G) ** 0.5
hbar = 6.582e-22                      # MeV*s

# MSW prefactor used in System_Nudecoupling
MSWPrefactor = 8.65785 * GF / (mW**2)

# ========= Oscillation configuration =========
# Change only this line to set the default behavior on import.
DEFAULT_IFOSC = False

# Module-level toggle. It is set consistently by set_ifOsc(DEFAULT_IFOSC) below.
ifOsc = None

# ========= Oscillation / PMNS placeholders =========
Dm21sq = None
Dm31sq = None

s12 = None
s23 = None
s13 = None

c12 = None
c23 = None
c13 = None

ds12 = None
ds23 = None
ds13 = None

dc12 = None
dc23 = None
dc13 = None

deltaCP = 0.0

# PMNS elements and mass-squared matrix components (deltaCP = 0, NO with m1=0)
Ue1 = None
Ue2 = None
Ue3 = None
Umu1 = None
Umu2 = None
Umu3 = None
Utau1 = None
Utau2 = None
Utau3 = None

m1 = 0.0
m2 = None
m3 = None

m11 = None
m12 = None
m13 = None
m22 = None
m23 = None
m33 = None


def _apply_mixing_parameters(_s12, _s23, _s13, _Dm21sq, _Dm31sq):
    """
    Compute all derived mixing, PMNS, and flavor-basis mass-squared quantities.
    Assumes deltaCP = 0 and normal ordering with m1 = 0.
    """
    global s12, s23, s13, c12, c23, c13
    global ds12, ds23, ds13, dc12, dc23, dc13
    global Dm21sq, Dm31sq
    global Ue1, Ue2, Ue3, Umu1, Umu2, Umu3, Utau1, Utau2, Utau3
    global m2, m3, m11, m12, m13, m22, m23, m33

    Dm21sq = float(_Dm21sq)
    Dm31sq = float(_Dm31sq)

    s12 = float(_s12)
    s23 = float(_s23)
    s13 = float(_s13)

    c12 = float(np.sqrt(max(0.0, 1.0 - s12**2)))
    c23 = float(np.sqrt(max(0.0, 1.0 - s23**2)))
    c13 = float(np.sqrt(max(0.0, 1.0 - s13**2)))

    # Double-angle combinations used in calcTransfers
    ds12 = float(2.0 * s12 * c12)
    ds23 = float(2.0 * s23 * c23)
    ds13 = float(2.0 * s13 * c13)

    dc12 = float(c12**2 - s12**2)
    dc23 = float(c23**2 - s23**2)
    dc13 = float(c13**2 - s13**2)

    # PMNS matrix elements for deltaCP = 0
    Ue1 = c12 * c13
    Ue2 = s12 * c13
    Ue3 = s13

    Umu1 = -s12 * c23 - c12 * s23 * s13
    Umu2 = c12 * c23 - s12 * s23 * s13
    Umu3 = s23 * c13

    Utau1 = s12 * s23 - c12 * c23 * s13
    Utau2 = -c12 * s23 - s12 * c23 * s13
    Utau3 = c23 * c13

    # Masses in normal ordering with m1 = 0
    global m1
    m2 = float(np.sqrt(max(0.0, Dm21sq)))
    m3 = float(np.sqrt(max(0.0, Dm31sq)))

    # Elements of m^2 in flavor basis
    m11 = m1**2 * Ue1**2 + m2**2 * Ue2**2 + m3**2 * Ue3**2
    m12 = m1**2 * Ue1 * Umu1 + m2**2 * Ue2 * Umu2 + m3**2 * Ue3 * Umu3
    m13 = m1**2 * Ue1 * Utau1 + m2**2 * Ue2 * Utau2 + m3**2 * Ue3 * Utau3
    m22 = m1**2 * Umu1**2 + m2**2 * Umu2**2 + m3**2 * Umu3**2
    m23 = m1**2 * Umu1 * Utau1 + m2**2 * Umu2 * Utau2 + m3**2 * Umu3 * Utau3
    m33 = m1**2 * Utau1**2 + m2**2 * Utau2**2 + m3**2 * Utau3**2


def set_ifOsc(flag: bool):
    """
    Switch oscillations on or off and recompute all derived constants.

    When oscillations are off, the mixing angles are set to zero so that
    transfer matrices reduce to the identity. The mass splittings are kept
    at their nominal values, though they are not used in that case.
    """
    global ifOsc
    ifOsc = bool(flag)

    if ifOsc:
        # NuFIT 5.2 (NO) values used in your previous file
        _apply_mixing_parameters(
            _s12=(0.303) ** 0.5,
            _s23=(0.572) ** 0.5,
            _s13=(0.02203) ** 0.5,
            _Dm21sq=7.41e-17,
            _Dm31sq=2.511e-15,
        )
    else:
        _apply_mixing_parameters(
            _s12=0.0,
            _s23=0.0,
            _s13=0.0,
            _Dm21sq=7.41e-17,
            _Dm31sq=2.511e-15,
        )


# Initialize the module consistently on import.
set_ifOsc(DEFAULT_IFOSC)
