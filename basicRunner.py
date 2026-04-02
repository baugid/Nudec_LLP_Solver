# copy paste from Solve_Nudecoupling.ipynb since this is easier to work with
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Distributions
from Constants import *
import Momentum_Grid
from System_Nudecoupling import *
from LLP_parameters import *

doPlots = False
useDecayProbabilities = True

z_0 = np.zeros(1)

z_0[0] = 1.00003

t_0 = np.array([0.])  # starting time will be ~1ms since BB but this is only used for decays so 0 is simpler

# compute stop point
xTimeData = np.loadtxt("scaleFactorTime.csv", delimiter=",")
timeIdx = np.searchsorted(xTimeData[:, 1], llp_lifetime * lifetimeFactor)

xOftSlope = (xTimeData[timeIdx, 0] - xTimeData[timeIdx - 1, 0]) / (xTimeData[timeIdx, 1] - xTimeData[timeIdx - 1, 1])

stopPoint = xOftSlope * (llp_lifetime * lifetimeFactor - xTimeData[timeIdx - 1, 1]) + xTimeData[timeIdx - 1, 0]

# compute maximum comoving momentum
if llp_twoNuDecayE + llp_twoNuDecayMu + llp_twoNuDecayTau > 0:
    ylimit = np.max([stopPoint * mmu / 2, 40, stopPoint * llp_mass / 2])
else:
    ylimit = np.max([stopPoint * mmu / 2, 40])

gridPoints = 201
Momentum_Grid.setupGrid(ylimit, gridPoints)
Distributions.initDistributions(llp_mass)

print(f"Grid limit: y_max={ylimit:.2f}\tn={gridPoints}\tstopPoint={stopPoint:.2f}")

if useDecayProbabilities:
    decProbabilities = np.loadtxt("decayProbs.csv", delimiter=',')[::-1]


    def decayHandler(T):
        """
        Computes the decay probability at a given temperature for all particles.
        The data is taken from the decProbabilites array
        :param T: The current temperature
        :return: A dictionary mapping a particle identifier (e.g. "pi") to the decay probability
        """
        idx = np.searchsorted(decProbabilities[:, 0], T)
        if idx == len(decProbabilities):
            return {"mu": decProbabilities[idx - 1, 1], "pi": decProbabilities[idx - 1, 2]}
        elif T < decProbabilities[0, 0]:
            return {"mu": decProbabilities[0, 1], "pi": decProbabilities[0, 2]}
        else:
            slopes = (decProbabilities[idx] - decProbabilities[idx - 1]) / (
                    decProbabilities[idx, 0] - decProbabilities[idx - 1, 0])
            vals = slopes * (T - decProbabilities[idx - 1, 0]) + decProbabilities[idx - 1]
            # assert vals[0] == T
            return {"mu": vals[1], "pi": vals[2]}
else:
    decayHandler = lambda T: {"mu": 1., "pi": 1.}

# setup the initial distributions
f_nue_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)
f_numu_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)
f_nutau_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)

sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

x_span = [0.1, 20]  # [0.0511, 35]

# setup llp args
llp_branchings = [llp_muonBranching, llp_pionBranching, llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau]
argList = [llp_abundance * (x_span[0] / me) ** 3, llp_lifetime, llp_mass, llp_branchings, stopPoint, decayHandler]

# RK45 or LDSODA might be better.
start = time.time()
sol = solve_ivp(System_Nudec, x_span, sys_values_0, args=argList, method='RK45', t_eval=None, atol=1e-8, rtol=1e-8)


def calcNeff(distri):
    rho1 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[:Momentum_Grid.n])
    rho2 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[Momentum_Grid.n:2 * Momentum_Grid.n])
    rho3 = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[2 * Momentum_Grid.n:3 * Momentum_Grid.n])
    rho4 = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * 1 / (np.exp(Momentum_Grid.gridVals / 1.0000) + 1))

    return ((11 / 4) ** (1 / 3) / distri[3 * Momentum_Grid.n]) ** 4 * (rho1 + rho2 + rho3) / rho4


Neff = calcNeff(sol.y[:, -1])

if doPlots:
    plt.plot(sol.t, [calcNeff(sol.y[:, k]) for k in range(len(sol.t))])
    plt.ylim(ymax=4, ymin=2.5)
    plt.xlabel("x in MeV")
    plt.ylabel("Neff")
    plt.show()
    plt.plot(sol.t, sol.y[-1, :])
    plt.xlabel("x in MeV")
    plt.ylabel("t in s")
    plt.show()

# output the time scalefactor relation (interesting for debug purposes)
# resArray = np.array([[t, y] for t, y in zip(sol.t, sol.y[-1, :])])
# np.savetxt("scaleTime.csv", resArray, delimiter=",")

print(f"N_eff={Neff}")

print(f"t_end={sol.y[-1, -1]:.4}s")
print(f"T_end={sol.y[-2, -1] / sol.t[-1] * me:.4}")
print(f"Steps={sol.nfev}")
print(f"Runtime: {time.time() - start:.4}s")
