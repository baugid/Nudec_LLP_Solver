import time
import os
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Distributions
from Constants import *
import Momentum_Grid
from System_Nudecoupling import *
from GlobalParameters import *
import sys

# Get command-line arguments
parameterIdx = int(sys.argv[1])  # index of the run
dir_script = sys.argv[2].rstrip('/')  # directory containing the scripts
dir_input = sys.argv[3]  # input data directory
dir_output = sys.argv[4]  # directory to place output
timeScaleFactorFile = os.path.join(dir_script, 'scaleFactorTime.csv')
finalTemperature = 0.025
maxXEnd = 1.0e5
gridInjectionSafety = 2.0
thermalCollisionUMax = 15.0
thermalCollisionYLimitCap = 120.0
llpDominationGridThreshold = 2.0

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


def estimateThermalCollisionGridLimit(gridStopPoint):
    rhoLLPStart = llp_mass * llp_abundance
    TStart = 1.00003 * me / initialX
    rhoRadStart = (np.pi ** 2 / 30.0) * 10.75 * TStart ** 4

    if rhoRadStart <= 0.0 or rhoLLPStart / rhoRadStart <= llpDominationGridThreshold:
        return 0.0

    TReheat = 0.7 / np.sqrt(llp_lifetime)
    zReheatEstimate = gridStopPoint * TReheat / me
    if zReheatEstimate <= 0.0:
        return 0.0

    return min(thermalCollisionYLimitCap, thermalCollisionUMax * zReheatEstimate)

# Compute stop point
if endInjection:
    injectionCutoffTime = llp_lifetime * lifetimeFactor
    xTimeData = np.loadtxt(timeScaleFactorFile, delimiter=",")
    timeIdx = np.searchsorted(xTimeData[:, 1], injectionCutoffTime)

    if timeIdx == 0:
        stopPoint = xTimeData[0, 0]
    elif timeIdx >= len(xTimeData):
        stopPoint = xTimeData[-1, 0]
    else:
        xOftSlope = (xTimeData[timeIdx, 0] - xTimeData[timeIdx - 1, 0]) / (
                xTimeData[timeIdx, 1] - xTimeData[timeIdx - 1, 1])
        stopPoint = xOftSlope * (injectionCutoffTime - xTimeData[timeIdx - 1, 1]) + xTimeData[timeIdx - 1, 0]
    gridStopPoint = stopPoint

    # Compute maximum comoving momentum
    limits = [SMComovingLimit]
    if llp_muonBranching > 0 or llp_pionBranching > 0:
        limits.append(gridInjectionSafety * gridStopPoint * mmu / (2 * me))
    if llp_twoNuDecayE + llp_twoNuDecayMu + llp_twoNuDecayTau > 0:
        limits.append(gridInjectionSafety * gridStopPoint * llp_mass / (2 * me))
    thermalCollisionLimit = estimateThermalCollisionGridLimit(gridStopPoint)
    if thermalCollisionLimit > 0:
        limits.append(thermalCollisionLimit)
    ylimit = np.max(limits)
else:
    injectionCutoffTime = np.inf
    stopPoint = finalX
    gridStopPoint = finalX
    limits = [SMComovingLimit]
    if llp_muonBranching > 0 or llp_pionBranching > 0:
        limits.append(gridInjectionSafety * gridStopPoint * mmu / (2 * me))
    if llp_twoNuDecayE + llp_twoNuDecayMu + llp_twoNuDecayTau > 0:
        limits.append(gridInjectionSafety * gridStopPoint * llp_mass / (2 * me))
    thermalCollisionLimit = estimateThermalCollisionGridLimit(gridStopPoint)
    if thermalCollisionLimit > 0:
        limits.append(thermalCollisionLimit)
    ylimit = np.max(limits)

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


def fermiDiracZeroMu(arg):
    return np.exp(-np.logaddexp(0.0, arg))


# Set up the initial distributions
z_0 = np.zeros(1)
z_0[0] = 1.00003

t_0 = np.array([0.])  # starting time will be ~1ms since BB but this is only used for decays so 0 is simpler

f_nue_0 = fermiDiracZeroMu(Momentum_Grid.gridVals / z_0[0])
f_numu_0 = fermiDiracZeroMu(Momentum_Grid.gridVals / z_0[0])
f_nutau_0 = fermiDiracZeroMu(Momentum_Grid.gridVals / z_0[0])

sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

x_span = [initialX, finalX]  # [0.0511, 35]

# setup LLP args
llp_branchings = [llp_muonBranching, llp_pionBranching, llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau]
argList = [
    llp_abundance * (x_span[0] / me) ** 3,
    llp_lifetime,
    llp_mass,
    llp_branchings,
    stopPoint,
    decayHandler,
    False,
    injectionCutoffTime,
    t_0[0],
]


def stopAtTfinal(x, state, *_args):
    return state[-2] / x * me - finalTemperature


stopAtTfinal.terminal = True
stopAtTfinal.direction = -1.0


def solveUntilTfinal():
    xs = []
    ys = []
    nfev = 0
    currentX = x_span[0]
    currentXEnd = x_span[1]
    currentState = sys_values_0.copy()

    while True:
        solPart = solve_ivp(
            System_Nudec,
            [currentX, currentXEnd],
            currentState,
            args=argList,
            method='RK45',
            t_eval=None,
            atol=1e-8,
            rtol=1e-8,
            events=stopAtTfinal,
        )
        nfev += solPart.nfev

        if xs:
            xs.append(solPart.t[1:])
            ys.append(solPart.y[:, 1:])
        else:
            xs.append(solPart.t)
            ys.append(solPart.y)

        if not solPart.success:
            raise RuntimeError(f"solve_ivp failed: {solPart.message}")

        reachedTfinal = len(solPart.t_events) > 0 and len(solPart.t_events[0]) > 0
        currentT = solPart.y[-2, -1] / solPart.t[-1] * me
        if reachedTfinal or currentT <= finalTemperature:
            return SimpleNamespace(t=np.concatenate(xs), y=np.concatenate(ys, axis=1), nfev=nfev)

        if currentXEnd >= maxXEnd:
            raise RuntimeError(
                f"Tfinal={finalTemperature} MeV was not reached before maxXEnd={maxXEnd}. "
                f"Final T={currentT:.10e} MeV."
            )

        currentX = solPart.t[-1]
        currentState = solPart.y[:, -1].copy()
        currentXEnd = min(maxXEnd, max(2.0 * currentXEnd, currentX + 1.0))
        print(f"Extending integration ceiling to x = {currentXEnd:.6e} to reach Tfinal = {finalTemperature:.6e} MeV")


# RK45 or LDSODA might be better.
start = time.time()
sol = solveUntilTfinal()


def calcNeff(distri):
    rho1 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[:Momentum_Grid.n])
    rho2 = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[Momentum_Grid.n:2 * Momentum_Grid.n])
    rho3 = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * distri[2 * Momentum_Grid.n:3 * Momentum_Grid.n])
    rho4 = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * fermiDiracZeroMu(Momentum_Grid.gridVals))

    return ((11 / 4) ** (1 / 3) / distri[3 * Momentum_Grid.n]) ** 4 * (rho1 + rho2 + rho3) / rho4


Neff = calcNeff(sol.y[:, -1])
if np.isfinite(injectionCutoffTime) and sol.y[-1, -1] - t_0[0] < injectionCutoffTime:
    print("WARNING: final time is before the LLP injection cutoff. N_eff is not a post-decay value.")

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
