import numpy as np
from Constants import *
import Momentum_Grid


def diracDeltaDistribution(position, width, index, height=1):  # index: 0=e, 1=mu, 2=tau
    def rebin(ps):
        lowerIdx = np.searchsorted(ps, position - width, side='right') - 1
        upperIdx = np.searchsorted(ps, position + width, side='right')

        # clamp to array limits
        lowerIdx = max(0, lowerIdx)
        upperIdx = min(upperIdx, len(ps) - 1)

        res = np.array([np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n)])
        # catch corner cases where width*2<bin width
        if lowerIdx == upperIdx:
            if upperIdx == len(ps) - 1:  # TODO this uses the width of the bin one step lower, breaks for log-space
                res[index, lowerIdx] = height / (ps[upperIdx] - ps[lowerIdx - 1])
            else:
                res[index, lowerIdx] = height / (ps[upperIdx + 1] - ps[lowerIdx])
        else:  # TODO is it correct how the length is computed? Should it be upperIdx+1?
            res[index, lowerIdx:upperIdx] = height / (ps[upperIdx] - ps[lowerIdx])
        return res

    return rebin


def trueDelta(position, flavour, particle_count=1):
    """
    Deposit a monochromatic source at physical momentum `position` using the two
    neighboring bins, in a way that preserves both the injected particle number
    and the injected energy in the discrete quadrature.

    If the line lies outside the momentum grid, raise an error instead of
    silently clipping it into the first/last bin.
    """
    def rebin(ps, x, _=None):
        if len(ps) < 2:
            raise ValueError("Momentum grid must contain at least two points for two-bin deposition.")

        if position < ps[0] or position > ps[-1]:
            raise ValueError(
                f"Injected momentum {position:.12e} MeV lies outside the grid range "
                f"[{ps[0]:.12e}, {ps[-1]:.12e}] MeV. Increase y_max."
            )

        res = np.array([np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n)])
        norm = x / me * particle_count

        # If the line lands exactly on a grid point, deposit all of it there.
        exact_idx = np.where(np.isclose(ps, position, rtol=1e-13, atol=1e-15))[0]
        if exact_idx.size > 0:
            idx = int(exact_idx[0])
            res[flavour, idx] = norm / Momentum_Grid.gridWeights[idx]
            return res

        right_idx = np.searchsorted(ps, position, side='right')
        left_idx = right_idx - 1

        if left_idx < 0 or right_idx >= len(ps):
            raise ValueError(
                f"Failed to bracket injected momentum {position:.12e} MeV inside the grid. "
                f"Grid range is [{ps[0]:.12e}, {ps[-1]:.12e}] MeV."
            )

        p_left = ps[left_idx]
        p_right = ps[right_idx]

        if p_right <= p_left:
            raise ValueError(
                f"Non-increasing momentum grid encountered around indices {left_idx}, {right_idx}."
            )

        alpha = (p_right - position) / (p_right - p_left)
        beta = (position - p_left) / (p_right - p_left)

        res[flavour, left_idx] = norm * alpha / Momentum_Grid.gridWeights[left_idx]
        res[flavour, right_idx] = norm * beta / Momentum_Grid.gridWeights[right_idx]
        return res

    return rebin


def muonDistribution(ps, x, decayProbabilities):
    if Momentum_Grid.debug_grid and ps[-1] < mmu / 2:
        print("Muon clipping")
    maxIdx = np.searchsorted(ps, mmu / 2, side='right')  # TODO this will cause problems if this is zero

    res_nue = 96 * (ps ** 2) * (1 - 2 * ps / mmu) / (mmu ** 3)
    res_nue[maxIdx:] = 0

    res_numu = 48 * (ps ** 2) * (1 - 4 * ps / (3 * mmu)) / (mmu ** 3)
    res_numu[maxIdx:] = 0

    return decayProbabilities["mu"] * np.array([res_nue, res_numu, np.zeros(Momentum_Grid.n)])


__pion_neutrino_energy = (mpi ** 2 - mmu ** 2) / (2 * mpi)


def pionDistribution(ps, x, decayProbabilities):
    if Momentum_Grid.debug_grid and ps[-1] < __pion_neutrino_energy:
        print("Pion clipping")
    res = trueDelta(__pion_neutrino_energy, 1, 1)(ps, x)
    res += muonDistribution(ps, x, decayProbabilities)
    return decayProbabilities["pi"] * res


# contains for each possible meson a function p->[df_nue,df_numu,df_nutau], where p contains the momenta for all bins
def initDistributions(mass=0):
    global getDistribution
    getDistribution = [muonDistribution, pionDistribution,
                       trueDelta(mass / 2, 0, 2),
                       trueDelta(mass / 2, 1, 2),
                       trueDelta(mass / 2, 2, 2)]
