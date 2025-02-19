"""
Central code for the simulation
"""

import os

import numpy as np
from scipy.integrate import solve_ivp

from GlobalParameters import *
from Constants import *
import Momentum_Grid
from Distributions import initDistributions
from System_Nudecoupling import System_Nudec


def calcNeff(sys_vals):
    """
    Calculates N_eff for a list containing the output of the solver at a given timepoint
    :param sys_vals: Values of the system of equations at a given point
    :return: N_eff
    """
    rho_nue = np.sum(Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * sys_vals[:Momentum_Grid.n])
    rho_numu = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * sys_vals[Momentum_Grid.n:2 * Momentum_Grid.n])
    rho_nutau = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * sys_vals[2 * Momentum_Grid.n:3 * Momentum_Grid.n])
    rho_thermal = np.sum(
        Momentum_Grid.gridWeights * Momentum_Grid.gridVals ** 3 * 1 / (np.exp(Momentum_Grid.gridVals / 1.0000) + 1))

    return ((11 / 4) ** (1 / 3) / sys_vals[3 * Momentum_Grid.n]) ** 4 * (rho_nue + rho_numu + rho_nutau) / rho_thermal


class Simulator:
    """
    Class managing the simulation and relevant data.
    """

    def __init__(self, mass, lifetime, density, lifetimeFactor, branchings, directNeutrinos):
        """

        :param mass: Mass of the LLP (in MeV)
        :param lifetime: Lifetime of the LLP (in s)
        :param density: Number density of the LLP (in MeV^3)
        :param lifetimeFactor: factor to stop injection
        :param branchings: branching ratios in the same order as in Distributions.initDistributions
        :param directNeutrinos: True if direct neutrino injections are present
        """
        self.mass = mass
        self.lifetime = lifetime
        self.density = density
        self.lifetimeFactor = lifetimeFactor
        self.stopPoint = finalX
        self.branchings = branchings
        self.directNeutrinos = directNeutrinos

        self.useDecayProbability = False
        self.decayHandler = lambda t: {"mu": 1., "pi": 1.}

        self.ylimit = minComovingLimit
        self.solution = None
        self.timeOffset = 0.

        self.calcMaximalComoving()

    def calcMaximalComoving(self):
        """
        From the stored data calculate the maximum comoving momentum necessary.
        This method usually should not be called directly unless internal variables have been manually edited.
        """
        if self.directNeutrinos:
            self.ylimit = np.max([self.stopPoint * mmu / 2, minComovingLimit, self.stopPoint * self.mass / 2])
        else:
            self.ylimit = np.max([self.stopPoint * mmu / 2, minComovingLimit])

    def limitInjection(self):
        """
        If the injection of LLPs should end before the end of the simulation,
        this method must be called.
        """
        xTimeData = np.loadtxt(os.path.join(os.path.dirname(__file__), 'scaleFactorTime.csv'), delimiter=",")
        timeIdx = np.searchsorted(xTimeData[:, 1], self.lifetime * self.lifetimeFactor)

        xOftSlope = (xTimeData[timeIdx, 0] - xTimeData[timeIdx - 1, 0]) / (
                xTimeData[timeIdx, 1] - xTimeData[timeIdx - 1, 1])

        self.stopPoint = (xOftSlope * (self.lifetime * self.lifetimeFactor - xTimeData[timeIdx - 1, 1])
                          + xTimeData[timeIdx - 1, 0])
        self.calcMaximalComoving()

    def initializeSolver(self):
        """
        Initialize the solver. Afterwards no changes influencing the grid or injected distributions should be made.
        Otherwise, the behavior might be unexpected.
        """
        Momentum_Grid.setupGrid(self.ylimit, binCount)
        initDistributions(self.mass)

        if informationDuringRun:
            print(f"Grid limit: y_max = {self.ylimit:.2f}MeV  n = {binCount}  stopPoint = {self.stopPoint:.2f}")

    def loadDecayProbabilities(self, filename):
        """
        Load the decay probabilities specified in a file.
        :param filename: Filename of the file to load.
        """
        decProbabilities = np.loadtxt(filename, delimiter=',', skiprows=1)

        with open(filename, "r") as f:
            self.timeOffset = float(f.readline())

        def handler(t):
            """
            Computes the decay probability at a given time for all particles.
            The data is taken from the decProbabilities array
            :param t: The current time (in the frame of the solver)
            :return: A dictionary mapping a particle identifier (e.g. "pi") to the decay probability
            """
            tReal = t + self.timeOffset
            idx = np.searchsorted(decProbabilities[:, 0], tReal)
            # Handle edge cases, where t is outside the range
            if idx == len(decProbabilities):
                return {"mu": decProbabilities[idx - 1, 1], "pi": decProbabilities[idx - 1, 2]}
            elif tReal < decProbabilities[0, 0]:
                return {"mu": decProbabilities[0, 1], "pi": decProbabilities[0, 2]}
            else:
                # Linearly interpolate in all columns
                slopes = (decProbabilities[idx] - decProbabilities[idx - 1]) / (
                        decProbabilities[idx, 0] - decProbabilities[idx - 1, 0])
                vals = slopes * (tReal - decProbabilities[idx - 1, 0]) + decProbabilities[idx - 1]
                # assert vals[0] == tReal
                return {"mu": vals[1], "pi": vals[2]}

        self.decayHandler = handler

    def simulate(self):
        """
        Run the simulation.
        """
        z_0 = np.array([1.00003])  # at the starting point a*T=1 does not hold perfectly
        t_0 = np.array([0.])  # internally time starts at 0
        f_nue_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0) + 1)
        f_numu_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0) + 1)
        f_nutau_0 = 1 / (np.exp(Momentum_Grid.gridVals / z_0) + 1)
        sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

        x_span = [initialX, finalX]

        # setup additional arguments
        argList = [self.density * (x_span[0] / me) ** 3, self.lifetime, self.mass, self.branchings, self.stopPoint,
                   self.decayHandler]

        # RK45 or LDSODA might be better.
        self.solution = solve_ivp(System_Nudec, x_span, sys_values_0, args=argList, method='RK45', t_eval=None,
                                  atol=1e-9, rtol=1e-9)

    def finalNeff(self):
        """
        Calculates N_eff at the latest analyzed time point.
        :return: The value for N_eff at the end.
        """
        return calcNeff(self.solution.y[:, -1])

    def exportTimeScaleFactor(self, filename):
        """
        Output the time scalefactor relation (interesting for debug purposes).
        This way scaleFactorTime.csv was created.
        """
        resArray = np.array([[t, y] for t, y in zip(self.solution.t, self.solution.y[-1, :])])
        np.savetxt(filename, resArray, delimiter=",")

    def getSolution(self):
        """
        Get the full scipy solution object.
        :return: The solution object of the system.
        """
        return self.solution

    def exportResults(self, filename):
        """
        Export the results to the file specified in filename.
        :param filename: The filename to write the results to.
        """
        Neff = self.finalNeff()

        # Extract data from the solution
        scaleFactor = self.solution.t / me
        times = self.solution.y[-1, :]
        temp = self.solution.y[-2, :] / scaleFactor

        momentum = Momentum_Grid.gridVals / scaleFactor[-1]
        nu_e = self.solution.y[:Momentum_Grid.n, -1]
        nu_mu = self.solution.y[Momentum_Grid.n:2 * Momentum_Grid.n, -1]
        nu_tau = self.solution.y[2 * Momentum_Grid.n:3 * Momentum_Grid.n, -1]

        with open(filename, "w") as f:
            # first repeat the input and add Neff
            f.write(
                f"{self.mass} {self.lifetime} {self.density} {' '.join(map(str, self.branchings))} {self.lifetimeFactor} {Neff}\n")
            f.write("\n")  # one blank line

            # Include the scale factor, time, temperature table
            for a, t, T in zip(scaleFactor, times, temp):
                f.write(f"{a} {t + self.timeOffset} {T}\n")
            f.write("\n\n")  # two blank lines

            # neutrino distributions at the end
            for p, fe, fmu, ftau in zip(momentum, nu_e, nu_mu, nu_tau):
                f.write(f"{p} {fe} {fmu} {ftau}\n")
