# Nu Decoupling Simple

This code evolves neutrino momentum distributions in the MeV plasma with LLP decays into electromagnetic particles, metastable particles, and neutrinos. The main entry point for production runs is `basicRunner_cli.py`.

The code was developed by Kensuke Akita, Gideon Baur, and Maksym Ovchynnikov. The physics setup is described in arXiv:2411.00892 and arXiv:2411.00931.

## Python Environment

If the system Python gives NumPy/SciPy/numba consistency or binary-compatibility errors, use a virtual environment. This avoids mixing packages from system Python, `~/.local`, and cluster modules.

```bash
cd /eos/user/o/ovchynni/Nu_Decoupling_Simple
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy numba
python -m py_compile basicRunner_cli.py
```

When the venv is active, install and launch with the same interpreter:

```bash
python -m pip install numpy scipy numba
python basicRunner_cli.py --help
```

## Single Runs

`basicRunner_cli.py` writes outputs under:

```text
/eos/user/o/ovchynni/Traditional/<output-folder>
```

The `--output-folder` argument is required and must be a simple folder name, not a path.

Example neutrinophilic run:

```bash
cd /eos/user/o/ovchynni/Nu_Decoupling_Simple
source .venv/bin/activate
python basicRunner_cli.py \
  --llp-mass 200 \
  --llp-lifetime 0.1 \
  --llp-abundance 4.73 \
  --llp-pion-branching 0 \
  --llp-muon-branching 0 \
  --llp-two-nu-decay-e 0.33333 \
  --llp-two-nu-decay-mu 0.33333 \
  --llp-two-nu-decay-tau 0.33334 \
  --nbins 301 \
  --Tstart 5.0 \
  --output-folder Neutrinophilic-uniform-final \
  --ifDebugging False
```

Example electromagnetic reheating run:

```bash
python basicRunner_cli.py \
  --llp-mass 50 \
  --llp-lifetime 0.2 \
  --llp-abundance 3156.955 \
  --llp-pion-branching 0 \
  --llp-muon-branching 0 \
  --llp-two-nu-decay-e 0 \
  --llp-two-nu-decay-mu 0 \
  --llp-two-nu-decay-tau 0 \
  --nbins 301 \
  --Tstart 5.0 \
  --output-folder EM-reheating-fixed \
  --ifDebugging False
```

Each run appends one row to `<output-folder>/output.txt` and writes distribution and thermodynamics snapshots for the generated run id.

## Batch Submissions

Batch helpers are in `Batch-submissions/`.

The example parameter files are:

```text
Batch-submissions/parameters-EM-reheating.txt
Batch-submissions/parameters-neutrinophilic-uniform.txt
```

Each non-comment row has 10 columns:

```text
llp_mass llp_lifetime llp_abundance llp_pionBranching llp_muonBranching llp_twoNuDecayE llp_twoNuDecayMu llp_twoNuDecayTau nbins T_start
```

Create and submit an EM-reheating batch:

```bash
cd /eos/user/o/ovchynni/Nu_Decoupling_Simple/Batch-submissions
python sub-creator.py parameters-EM-reheating.txt EM-reheating-fixed
condor_submit condor_basicRunner_cli_EM-reheating-fixed.sub
```

Create and submit a neutrinophilic batch:

```bash
cd /eos/user/o/ovchynni/Nu_Decoupling_Simple/Batch-submissions
python sub-creator.py parameters-neutrinophilic-uniform.txt Neutrinophilic-uniform-final
condor_submit condor_basicRunner_cli_Neutrinophilic-uniform-final.sub
```

`sub-creator.py` validates the parameter file, creates/updates the run wrapper and log directory inside `Batch-submissions/`, and writes the Condor `.sub` file to the directory from which the script is launched. It does not create or require a manifest file.
