"""
Here are many of the options that should usually be kept consistent across all (mass,lifetime) points
"""

binCount = 301  # number of bins for the grid. Runtime quickly grows with this value(default: 301)
initialX = 0.1  # starting point of the integration (default: 0.1) (This is the *only* place to change the starting temperature)
finalX = 35  # This gives the end point of the integration, usually it only lightly affects runtime (default: 35)
parameterFile = "parameters.csv"  # file that lists all sets of parameters
useDecayProbabilities = True  # if set to false decayProbabilities referenced in parameterFile will be ignored
endInjection = True  # cuts the injection at a given multiple of the lifetime and should be turned on for LLPs
informationDuringRun = True  # if turned off no command line output will occur at any time, making crashes difficult to detect
outputFileTemplate = "result_{idx}.txt"  # pattern for the result filename from the calculation. idx is the number of the parameter set
minComovingLimit = 40  # in MeV lowest maximum comoving momentum, that is used regardless of LLP lifetime (default: 40)
printRate = 60  # minimum delay between status messages in seconds if informationDuringRun is true (default: 60)
