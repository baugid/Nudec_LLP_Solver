"""
Parameter file for the SingleShot.py script.
"""

llp_mass = 900  # mass in MeV
llp_lifetime = 0.05  # lifetime in s
llp_density = 0.790653  # initial abundance in MeV^3
llp_muonBranching = 0.0  # avg number of primary muons per LLP decay
llp_pionBranching = 1.434  # avg number of primary pions per LLP decay (x->pi+ pi- produces two pions)
llp_twoNuDecayE = 0.  # branching to two nu decay
llp_twoNuDecayMu = 0.
llp_twoNuDecayTau = 0.
llp_lifetimeFactor = 10  # cutoff for LLP injection at lifetime*lifetimeFactor.

useDecayProbabilities = True # if true the file specified in the next line is loaded
llp_probabilityPath = "decayProbabilities.csv" # path to the decay probabilities

outputFile = "results.txt" # path to the output file. Set to None to ignore