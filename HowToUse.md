To use this code a couple of files are of interest that will be explained subsequently.
Furthermore, it is important to note how the data formats are structured.

All dependencies are specified in `requirements.txt` and can be installed via
```batch
pip install -r requirements.txt
```
**Important:** This code requires a modern version of python3. It is incompatible with python2.

# Data formats

## Input
Let's first discuss the different formats for data to interact with this code.
When decay probabilities deviating from one should be taken into account, a file tabulating them is necessary to be
provided.
The first line of this file contains the time in seconds, when the starting temperature is reached.
With the variable names from `GlobalParameters.py` this temperature is given by

```python
T = 1.00003 * me / initialX
```

The subsequent lines all have the same form and contain in csv format the columns

```
t in s, P(muon decays), P(pion decays)
```

It is required that these are sorted in order of increasing time.
Furthermore, it is recommended that the values reach from a bit earlier than the starting point to the point of last
injection or a point, where all probabilities are sufficiently close to 1.

For batch runs the parameters have to be provided in a csv file like the provided `parameters.csv`.
See the sample file and `LLP_parameters.py` for details about the order and meaning of values.
In the last column for the file with decay probabilities the magic value `None` can be used to ignore it.
Note that the internal numbering of the lines starts at zero for the first line, which can be used as a header.

## Output

The output file consists of three sections. Its first line contains the input parameters separated by spaces in the same
order as everywhere else. After a blank line the relation between scale factor, time and temperature is given in the
format

```
a t(s) T(MeV)
```

Following two blank lines the distribution functions at the last time point are given.
The format is similar

```
p(MeV) f_nue(MeV) f_numu(MeV) f_nutau(MeV)
```

# Important files

## SingleShot.py

Use this script for single shot local calculations.
The parameters for the BSM particle are loaded from `LLP_parameters.py` and have to be set as documented there.
Further parameters in `GlobalParameters.py` are respected.

## BatchLauncher.py

This script is written to work with batches of cluster jobs.
Many global parameters across all runs are set in `GlobalParameters.py` and are documented there.
This includes the file name of the list of parameters.
To run this script, it is necessary to provide an index as a command line parameter as well as a folder for input and
output files

```bash
python BatchLauncher.py idx inputDirectory outputDirectory
```

This loads the content from the `idx+1` line (line 1 is assumed to be a header) of the parameter file specified.
It is important to note that the parameter file is also resolved relative to `inputDirectory`. 
The content of each line is similar to `LLP_parameters.py` and is documented there in more detail.
The formatting should be

```
mass, lifetime, numberDensity, muonBranching, pionBranching, lifetimeFactor, decayFile
```

as is also seen in the sample file `parameters.csv`.
Between `pionBranching` and `lifetimeFactor` the branching ratios for direct decays into pairs of neutrinos can be provided.
If they are present, the code takes into account the potentially higher maximal comoving momentum.
**Note:** Providing three zeros as branchings also counts as present, which might have unintended effects.

All values should be given in powers of MeV. The lifetime factor can usually be safely set to 5 or 10.
decayFile references the file where the decay probabilities are tabulated as described above.
If this is not necessary for one line set it to `None`.

## GlobalParameters.py

This file contains all parameters kept constant across multiple runs.
Most are set to safe defaults that sometimes correspond to long runtimes.
It might for example be safe to reduce `binCount` to 201 or even further.
Note that it must be an odd number.

# Extending the code

To add new decay modes the most interesting file is `Distributions.py`.
At the bottom a list called `getDistribution` is initialized.
Your custom distributions must be appended there. For details on normalization see the accompanying papers.
In that case the launch scripts must be modified to get the additional parameters and pass them along to the constructor
of the `Simulator` object.
Note that the rules for the maximum comoving momentum probably have to be adapted in `Simulator.calcMaximalComoving` in `Core.py`.
To catch errors the flag `debug_grid` hidden in `Momentum_Grid.py` might proof useful.
If it is necessary to add additional decay probabilities the function called `handler` defined in
`Simulator.loadDecayProbabilities` has to be extended accordingly.
