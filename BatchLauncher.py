"""
This is the starting script when running multiple input parameters in a batch.
Via command line the index into the parameter file (starting from 0 for the header line),
the folder, which contains the input parameter file and is used as a starting point to resolve the decay probability files,
as well as a folder into which the outputs should be generated has to be provided.
"""

import os
import sys
import time

from Core import Simulator
from GlobalParameters import *

# Get command-line arguments
if len(sys.argv) != 4:
    print("Usage: python BatchLauncher.py idx inputDirectory outputDirectory")
    sys.exit(1)

parameterIdx = int(sys.argv[1])  # index of the run
dir_input = sys.argv[2]  # input data directory
dir_output = sys.argv[3]  # directory to place output

# Load parameters
parameters_exist = False
directNeutrinos = False
branchings = []
with open(os.path.join(dir_input, parameterFile), "r") as f:
    for i, line in enumerate(f):
        if i == parameterIdx:
            parameters_exist = True
            splitted = line.split(',')

            numbersFromRow = [float(x) for x in splitted[:-1]]
            # If 6 numbers are present, direct decays to neutrinos are inactive
            if len(numbersFromRow) == 6:
                llp_mass, llp_lifetime, llp_density, llp_muonBranching, llp_pionBranching, llp_lifetimeFactor = numbersFromRow
                branchings = [llp_muonBranching, llp_pionBranching, 0, 0, 0]
                directNeutrinos = False
            else:
                llp_mass, llp_lifetime, llp_density, llp_muonBranching, llp_pionBranching, llp_twoNuDecayE, llp_twoNuDecayMu, llp_twoNuDecayTau, llp_lifetimeFactor = numbersFromRow
                branchings = [llp_muonBranching, llp_pionBranching, llp_twoNuDecayE, llp_twoNuDecayMu,
                              llp_twoNuDecayTau]
                directNeutrinos = True
            probabilityFile = splitted[-1].strip().strip('"')  # Strip surrounding quotes or spaces
            break

if not parameters_exist:
    print(f"There is no {parameterIdx}-th row in the parameters file. Terminating the output.")
    sys.exit(1)

simulation = Simulator(llp_mass, llp_lifetime, llp_density, llp_lifetimeFactor, branchings, directNeutrinos)

if endInjection:
    simulation.limitInjection()

# Check for a decay probabilities file and load it
if useDecayProbabilities and probabilityFile != "None":
    probabilityFilePath = os.path.join(dir_input, probabilityFile)
    if not os.path.isfile(probabilityFilePath):
        print(f"The decay probabilities file '{probabilityFilePath}' does not exist. Terminating the output.")
        sys.exit(1)

    simulation.loadDecayProbabilities(probabilityFilePath)

simulation.initializeSolver()

start = time.time()
simulation.simulate()
end = time.time()

if informationDuringRun:
    print(f"Runtime: {end - start:.1f}s.")

outputFile = os.path.join(dir_output, outputFileTemplate.format(idx=parameterIdx))
simulation.exportResults(outputFile)
