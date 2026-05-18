#!/usr/bin/env python3
from pathlib import Path
import argparse
import stat
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR = Path("/eos/user/o/ovchynni/Nu_Decoupling_Simple")
RUNNER = BASE_DIR / "basicRunner_cli.py"
OUTPUT_ROOT = Path("/eos/user/o/ovchynni/Traditional")

JOB_FLAVOUR = "testmatch"
ACCOUNTING_GROUP = "group_u_BE.ABP.NORMAL"


def output_folder_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("output_folder must not be empty")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise argparse.ArgumentTypeError("output_folder must be a simple folder name, not a path")
    return cleaned


def resolve_input_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Condor wrapper and submit file for basicRunner_cli.py. "
            "The runner outputs will be written under /eos/user/o/ovchynni/Traditional/OUTPUT_FOLDER."
        )
    )
    parser.add_argument(
        "parameters_file",
        type=resolve_input_path,
        help="Path to parameters.txt with 10 columns, where the last column is T_start",
    )
    parser.add_argument(
        "output_folder",
        type=output_folder_name,
        help="Name of the output subfolder XXX under /eos/user/o/ovchynni/Traditional/XXX",
    )
    return parser.parse_args()


def validate_parameters_file(param_file: Path) -> int:
    """
    Expect each non-empty, non-comment line in parameters.txt to contain exactly 10 columns:
      llp_mass llp_lifetime llp_abundance llp_pionBranching llp_muonBranching
      llp_twoNuDecayE llp_twoNuDecayMu llp_twoNuDecayTau nbins T_start
    """
    n_jobs = 0

    with param_file.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 10:
                raise ValueError(
                    f"{param_file}:{lineno}: expected 10 columns, got {len(parts)}\n"
                    f"Line: {line}"
                )

            try:
                nbins = int(parts[8])
            except ValueError as exc:
                raise ValueError(
                    f"{param_file}:{lineno}: nbins must be an integer\n"
                    f"Line: {line}"
                ) from exc

            if nbins < 3:
                raise ValueError(
                    f"{param_file}:{lineno}: nbins must be >= 3\n"
                    f"Line: {line}"
                )

            try:
                t_start = float(parts[9])
            except ValueError as exc:
                raise ValueError(
                    f"{param_file}:{lineno}: T_start must be a number\n"
                    f"Line: {line}"
                ) from exc

            if t_start <= 0:
                raise ValueError(
                    f"{param_file}:{lineno}: T_start must be > 0\n"
                    f"Line: {line}"
                )

            n_jobs += 1

    return n_jobs


def ensure_output_dirs(output_folder: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_ROOT / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_run_sh(run_sh: Path, output_folder: str):
    content = f"""#!/bin/bash
source {BASE_DIR}/.venv/bin/activate
python {RUNNER} --llp-mass "$1" --llp-lifetime "$2" --llp-abundance "$3" --llp-pion-branching "$4" --llp-muon-branching "$5" --llp-two-nu-decay-e "$6" --llp-two-nu-decay-mu "$7" --llp-two-nu-decay-tau "$8" --nbins "$9" --Tstart "${{10}}" --output-folder "{output_folder}"
"""

    run_sh.write_text(content, encoding="utf-8")

    mode = run_sh.stat().st_mode
    run_sh.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_sub_file(sub_file: Path, run_sh: Path, logs_dir: Path, param_file: Path):
    content = f"""executable = {run_sh}
arguments = $(llp_mass) $(llp_lifetime) $(llp_abundance) $(llp_pion_branching) $(llp_muon_branching) $(llp_two_nu_decay_e) $(llp_two_nu_decay_mu) $(llp_two_nu_decay_tau) $(nbins) $(T_start)
transfer_output_files = ""
request_cpus = 1
+JobFlavour = "{JOB_FLAVOUR}"
log = {logs_dir}/basicRunner_$(Process).log
error = {logs_dir}/basicRunner_$(Process).err
output = /dev/null
+AccountingGroup = "{ACCOUNTING_GROUP}"

queue llp_mass, llp_lifetime, llp_abundance, llp_pion_branching, llp_muon_branching, llp_two_nu_decay_e, llp_two_nu_decay_mu, llp_two_nu_decay_tau, nbins, T_start from {param_file}
"""
    sub_file.write_text(content, encoding="utf-8")


def main():
    args = parse_args()

    param_file = args.parameters_file
    output_folder = args.output_folder
    submission_dir = Path.cwd().resolve()
    run_sh = SCRIPT_DIR / f"run_{output_folder}.sh"
    sub_file = submission_dir / f"condor_basicRunner_cli_{output_folder}.sub"
    logs_dir = SCRIPT_DIR / "logs" / output_folder

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")
    if not RUNNER.exists():
        raise FileNotFoundError(f"Runner script not found: {RUNNER}")
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    ensure_output_dirs(output_folder)
    n_jobs = validate_parameters_file(param_file)

    logs_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_sh(run_sh, output_folder)
    write_sub_file(sub_file, run_sh, logs_dir, param_file)

    print(f"Read parameters from: {param_file}")
    print(f"Output root ensured:  {OUTPUT_ROOT}")
    print(f"Output folder ready:  {OUTPUT_ROOT / output_folder}")
    print(f"Created run wrapper:  {run_sh}")
    print(f"Created submit file:  {sub_file}")
    print(f"Log directory:        {logs_dir}")
    print(f"Found {n_jobs} jobs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
