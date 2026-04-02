import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Distributions
from Constants import *
import Momentum_Grid
from System_Nudecoupling import *
from globalParameters import *
import sys

# Get command-line arguments
parameterIdx = int(sys.argv[1])  # index of the run
dir_script = sys.argv[2].rstrip('/')  # directory containing the scripts
dir_input = sys.argv[3]  # input data directory
dir_output = sys.argv[4]  # directory to place output
timeScaleFactorFile = os.path.join(dir_script, 'scaleFactorTime.csv')

# Set the parameter file and output file template based on the provided LLP name
parameterFile = os.path.join(dir_input, "parameters.csv")

# Load parameters
parameters_exist = False
with open(parameterFile, "r") as f:
    for i, line in enumerate(f):
        if i == parameterIdx:
            parameters_exist = True
            splitted = line.split(',')

            numbersFromRow = [float(x) for x in splitted[:-1]]
            if len(numbersFromRow) == 6:
                llp_mass, llp_lifetime, llp_abundance, llp_pionBranching, llp_muonBranching, lifetimeFactor = numbersFromRow
                llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau = 0, 0, 0
            else:
                llp_mass, llp_lifetime, llp_abundance, llp_pionBranching, llp_muonBranching, llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau, lifetimeFactor = numbersFromRow

            probabilityFile = splitted[-1].strip().strip('"')  # Strip surrounding quotes
            break

if not parameters_exist:
    print("There is no ith row in the parameters file. Terminating the output.")
    sys.exit(1)

# Check if the decay probabilities file exists if specified
if useDecayProbabilities and probabilityFile != "None":
    probabilityFilePath = os.path.join(dir_input, probabilityFile)
    if not os.path.isfile(probabilityFilePath):
        print(f"The decay probabilities file '{probabilityFile}' does not exist. Terminating the output.")
        sys.exit(1)

# Compute stop point
if endInjection:
    xTimeData = np.loadtxt(timeScaleFactorFile, delimiter=",")
    timeIdx = np.searchsorted(xTimeData[:, 1], llp_lifetime * lifetimeFactor)

    xOftSlope = (xTimeData[timeIdx, 0] - xTimeData[timeIdx - 1, 0]) / (
            xTimeData[timeIdx, 1] - xTimeData[timeIdx - 1, 1])

    stopPoint = xOftSlope * (llp_lifetime * lifetimeFactor - xTimeData[timeIdx - 1, 1]) + xTimeData[timeIdx - 1, 0]

    # Compute maximum comoving momentum
    if llp_twoNuDecayE + llp_twoNuDecayMu + llp_twoNuDecayTau > 0:
        ylimit = np.max([stopPoint * mmu / 2, 40, stopPoint * llp_mass / 2])
    else:
        ylimit = np.max([stopPoint * mmu / 2, 40])
else:
    ylimit = SMComovingLimit
    stopPoint = finalX

gridPoints = binCount
Momentum_Grid.setupGrid(ylimit, gridPoints)
Distributions.initDistributions(llp_mass)

if useDecayProbabilities and probabilityFile != "None":
    decProbabilities = np.loadtxt(probabilityFilePath, delimiter=',')[::-1]

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

# Set up the initial distributions
z_0 = np.zeros(1)
z_0[0] = 1.00003

t_0 = np.array([0.])  # starting time will be ~1ms since BB but this is only used for decays so 0 is simpler

f_nue_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)
f_numu_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)
f_nutau_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0[0]) + 1)

sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

x_span = [initialX, finalX]  # [0.0511, 35]

# setup LLP args
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

scaleFactor = sol.t / me
times = sol.y[-1, :]
temp = sol.y[-2, :] / scaleFactor
momentum = Momentum_Grid.gridVals / scaleFactor[-1]
nu_e = sol.y[:Momentum_Grid.n, -1]
nu_mu = sol.y[Momentum_Grid.n:2 * Momentum_Grid.n, -1]
nu_tau = sol.y[2 * Momentum_Grid.n:3 * Momentum_Grid.n, -1]

outputFileTemplate = os.path.join(dir_output, outputFileTemplate.format(idx=parameterIdx))
with open(outputFileTemplate, "w") as f:
    f.write(" ".join(splitted[:-1]) + " " + str(Neff))  # Replace comma with space
    f.write("\n\n")
    for a, t, T in zip(scaleFactor, times, temp):
        f.write(f"{a} {t} {T}\n")  # Replace commas with spaces
    f.write("\n\n")
    for p, fe, fmu, ftau in zip(momentum, nu_e, nu_mu, nu_tau):
        f.write(f"{p} {fe} {fmu} {ftau}\n")  # Replace commas with spaces
