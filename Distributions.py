"""
In this file distributions of a particle from a decay are implemented.
It has to be changed to add custom decay modes.
For this purpose a method with the signature
(momenta, x, decayProbabilities) -> 3 x len(momenta) numpy array
has to be added to getDistribution at the end of this file.
In the return value the first index corresponds to the flavor, while the latter indices give the momentum distribution.
The values will later be scaled according to the amount of LLPs that decayed.
Note that you have to ensure that the branching_fractions parameter in System_Nudecoupling.py has appropriate length.
It might also be important to adapt the code for decay probabilities in Core.py.
"""
import numpy as np
from Constants import *
import Momentum_Grid


def deltaDistribution(position, flavour, particle_count=1):
    """
    Generates a function that behaves as a delta distribution under numerical integration.
    :param position: momentum at which the delta distribution appears.
    :param flavour: flavor of the injected neutrino.
    :param particle_count: Number of particles to inject. Please ensure consistency with the branching fraction.
    :return: A function to add to getDistribution
    """


    def distribution(ps, x, _=None):
        idx = np.searchsorted(ps, position, side='right') - 1
        idx = max(0, idx)
        if ps[idx] == 0:
            idx += 1

        res = np.array([np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n), np.zeros(Momentum_Grid.n)])
        res[flavour, idx] = x / me * particle_count / (Momentum_Grid.gridWeights[idx])
        return res

    return distribution


def muonDistribution(ps, x, decayProbabilities):
    """
    Momentum distribution of injected muons with corresponding decay probabilities.
    :param ps: momentum values at the bin points.
    :param x: current value of x ( = a*me)
    :param decayProbabilities: A dictionary containing the loaded decay probabilities.
    :return: The normalized distribution of injected pions.
    """
    if Momentum_Grid.debug_grid and ps[-1] < mmu / 2:
        print("Muon clipping")
    maxIdx = np.searchsorted(ps, mmu / 2, side='right')

    res_nue = 96 * (ps ** 2) * (1 - 2 * ps / mmu) / (mmu ** 3)
    res_nue[maxIdx:] = 0

    res_numu = 48 * (ps ** 2) * (1 - 4 * ps / (3 * mmu)) / (mmu ** 3)
    res_numu[maxIdx:] = 0

    return decayProbabilities["mu"] * np.array([res_nue, res_numu, np.zeros(Momentum_Grid.n)])


__pion_neutrino_energy = (mpi ** 2 - mmu ** 2) / (2 * mpi)


def pionDistribution(ps, x, decayProbabilities):
    """
    Same for pions as for muons. Note the modularity of the different functions.
    """
    if Momentum_Grid.debug_grid and ps[-1] < __pion_neutrino_energy:
        print("Pion clipping")
    res = deltaDistribution(__pion_neutrino_energy, 1, 1)(ps, x)
    res += muonDistribution(ps, x, decayProbabilities)
    return decayProbabilities["pi"] * res

def initDistributions(mass=0):
    """
    Set up the distributions. This must be called before the system of equations can be solved.
    :param mass: Mass of the LLP. Only necessary when X-> nu nu channels are in use.
    """
    global getDistribution
    getDistribution = [muonDistribution, pionDistribution,
                       deltaDistribution(mass / 2, 0, 2),
                       deltaDistribution(mass / 2, 1, 2),
                       deltaDistribution(mass / 2, 2, 2)]
