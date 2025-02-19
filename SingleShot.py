"""
Single shot evaluation script.
This file does not require any further command line arguments.
All parameters are set in LLP_Parameters.py and GlobalParameters.py
"""

import time

import Core
from LLP_parameters import *
from GlobalParameters import *
from Constants import *

branchings = [llp_muonBranching, llp_pionBranching, llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau]
directNeutrinos = llp_twoNuDecayE + llp_twoNuDecayMu + llp_twoNuDecayTau > 0
simulation = Core.Simulator(llp_mass, llp_lifetime, llp_density,
                            llp_lifetimeFactor, branchings, directNeutrinos)
# Compute the stop point
if endInjection:
    simulation.limitInjection()

if useDecayProbabilities:
    simulation.loadDecayProbabilities(llp_probabilityPath)

simulation.initializeSolver()
start = time.time()
simulation.simulate()
end = time.time()

Neff = simulation.finalNeff()

# simulation.exportTimeScaleFactor("scaleTime.csv")

print(f"N_eff={Neff}")

print(f"t_end={simulation.getSolution().y[-1, -1]:.4}s")
print(f"T_end={simulation.getSolution().y[-2, -1] / simulation.getSolution().t[-1] * me:.4}")
print(f"Steps={simulation.getSolution().nfev}")
print(f"Runtime: {end - start:.1f}s")

if outputFile is not None:
    simulation.exportResults(outputFile)
