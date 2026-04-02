#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

def usage():
    print(
        "Usage:\n"
        "  python3 test.py <Tnue> <Tnumu> <Tnutau> <TEM> <ifOsc> <Tfinal>\n\n"
        "Arguments:\n"
        "  Tnue    : initial ν_e temperature in MeV (e.g. 3.0)\n"
        "  Tnumu   : initial ν_μ temperature in MeV (e.g. 3.0)\n"
        "  Tnutau  : initial ν_τ temperature in MeV (e.g. 3.0)\n"
        "  TEM     : initial photon/e± temperature in MeV (e.g. 3.5)\n"
        "  ifOsc   : neutrino oscillations flag: true/false (1/0, yes/no, on/off)\n"
        "  Tfinal  : stop when T_γ reaches this MeV value (e.g. 2.0)\n\n"
        "Example:\n"
        "  python3 test.py 3.0 3.0 3.0 3.5 true 2.0\n"
    )

def parse_bool(token: str) -> bool:
    t = token.strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ValueError("ifOsc must be true/false (or 1/0, yes/no, on/off)")

# ----- CLI -----
if len(sys.argv) != 7:
    usage()
    sys.exit(1)

try:
    T_nue_init   = float(sys.argv[1])
    T_numu_init  = float(sys.argv[2])
    T_nutau_init = float(sys.argv[3])
    T_em_init    = float(sys.argv[4])
    ifOsc_flag   = parse_bool(sys.argv[5])
    T_em_final   = float(sys.argv[6])

    if min(T_nue_init, T_numu_init, T_nutau_init, T_em_init, T_em_final) <= 0.0:
        raise ValueError("All temperatures must be positive.")
    if not (T_em_init > T_em_final):
        raise ValueError("Require TEM (initial) > Tfinal.")
except Exception as e:
    print(f"Error: {e}\n")
    usage()
    sys.exit(1)

# ----- Set oscillations before importing system -----
import Constants
Constants.set_ifOsc(ifOsc_flag)
from Constants import me

# Now safe to import modules that depend on Constants
import Distributions
import Momentum_Grid
from System_Nudecoupling import System_Nudec

# ----- Initial conditions -----
z_0 = np.array([1.0])                # choose z0=1 so x0 controls Tγ
x0  = z_0[0] * me / T_em_init
t_0 = np.array([0.0])

# Momentum grid
ylimit = 40.0
gridPoints = 201
Momentum_Grid.setupGrid(ylimit, gridPoints)
Distributions.initDistributions(0.0)

y = Momentum_Grid.gridVals
w = Momentum_Grid.gridWeights
n = Momentum_Grid.n

print(f"[init] y_max={ylimit:.2f}  n={gridPoints}  x0={x0:.6f}  "
      f"T_gamma0={z_0[0]/x0*me:.3f} MeV  (ifOsc={ifOsc_flag}, Tfinal={T_em_final} MeV)")

def fFD_ratio(Tnu, Tem):
    ratio = Tnu / Tem
    return 1.0 / (np.exp(y / ratio) + 1.0)

f_nue_0   = fFD_ratio(T_nue_init,   T_em_init)
f_numu_0  = fFD_ratio(T_numu_init,  T_em_init)
f_nutau_0 = fFD_ratio(T_nutau_init, T_em_init)

sys_values_0 = np.concatenate((f_nue_0, f_numu_0, f_nutau_0, z_0, t_0))

# ----- LLP OFF -----
llp_abundance0      = 0.0
llp_lifetime_dummy  = 1.0
llp_mass_dummy      = 1.0
llp_branchings_zero = [0.0, 0.0, 0.0, 0.0, 0.0]
stopPoint_dummy     = x0
decayHandler_dummy  = lambda T: {"mu": 0.0, "pi": 0.0}

argList = [llp_abundance0, llp_lifetime_dummy, llp_mass_dummy,
           llp_branchings_zero, stopPoint_dummy, decayHandler_dummy]

# ----- Stop when Tγ hits T_em_final -----
def stop_at_Tfinal(x, state, *_args):
    z = state[3*n]
    T_gamma = z / x * me
    return T_gamma - T_em_final

stop_at_Tfinal.terminal  = True
stop_at_Tfinal.direction = -1.0

# Integrate
x_span = [x0, 20.0]
print(f"Progress: x = {x0:.5f}")
start = time.time()
sol = solve_ivp(System_Nudec, x_span, sys_values_0,
                args=argList, method='RK45',
                atol=1e-8, rtol=1e-8,
                events=stop_at_Tfinal)
elapsed = time.time() - start
print(f"[solve] status={sol.status}  steps={sol.nfev}  runtime={elapsed:.3f}s")

# ----- Build Δρ_ν(T) and per-flavor Δρ_ν,α(T) tables -----
# FD y^3 moment at unit temperature (same grid)
rhoFD_oneflavor = np.sum(w * y**3 * (1.0 / (np.exp(y) + 1.0)))

Ts = []
deltas_tot = []
deltas_e = []
deltas_mu = []
deltas_tau = []

for k in range(len(sol.t)):
    x_k = sol.t[k]
    state = sol.y[:, k]
    z_k = state[3*n]
    T_gamma_k = z_k / x_k * me

    f_nue_k   = state[:n]
    f_numu_k  = state[n:2*n]
    f_nutau_k = state[2*n:3*n]

    # comoving energy integrals S_alpha = ∫ y^3 f_alpha dy
    S_e   = np.sum(w * y**3 * f_nue_k)
    S_mu  = np.sum(w * y**3 * f_numu_k)
    S_tau = np.sum(w * y**3 * f_nutau_k)
    S_tot = S_e + S_mu + S_tau

    # Ratios to equilibrium (per your convention); EM cancels
    ratio_tot = (S_tot / (z_k**4)) / (3.0 * rhoFD_oneflavor)
    ratio_e   = (S_e   / (z_k**4)) / (1.0 * rhoFD_oneflavor)
    ratio_mu  = (S_mu  / (z_k**4)) / (1.0 * rhoFD_oneflavor)
    ratio_tau = (S_tau / (z_k**4)) / (1.0 * rhoFD_oneflavor)

    delta_tot = (ratio_tot - 1.0) * 100.0
    delta_e   = (ratio_e   - 1.0) * 100.0
    delta_mu  = (ratio_mu  - 1.0) * 100.0
    delta_tau = (ratio_tau - 1.0) * 100.0

    Ts.append(T_gamma_k)
    deltas_tot.append(delta_tot)
    deltas_e.append(delta_e)
    deltas_mu.append(delta_mu)
    deltas_tau.append(delta_tau)

Ts = np.array(Ts)
deltas_tot = np.array(deltas_tot)
deltas_e   = np.array(deltas_e)
deltas_mu  = np.array(deltas_mu)
deltas_tau = np.array(deltas_tau)

# Temperature grid (descending)
T_grid = np.linspace(T_em_init, T_em_final, 151)

# Interpolate to grid (need Ts ascending for np.interp)
sort_idx = np.argsort(Ts)
Ts_sorted     = Ts[sort_idx]
d_tot_sorted  = deltas_tot[sort_idx]
d_e_sorted    = deltas_e[sort_idx]
d_mu_sorted   = deltas_mu[sort_idx]
d_tau_sorted  = deltas_tau[sort_idx]

Tmin, Tmax = Ts_sorted[0], Ts_sorted[-1]
T_query = np.clip(T_grid, Tmin, Tmax)

delta_tot_on_grid = np.interp(T_query, Ts_sorted, d_tot_sorted)
delta_e_on_grid   = np.interp(T_query, Ts_sorted, d_e_sorted)
delta_mu_on_grid  = np.interp(T_query, Ts_sorted, d_mu_sorted)
delta_tau_on_grid = np.interp(T_query, Ts_sorted, d_tau_sorted)

# ----- Write results to tests/ next to this script -----
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "tests")
os.makedirs(out_dir, exist_ok=True)

suffix = "with-osc" if ifOsc_flag else "no-osc"
fname = (
    f"Kensuke_TEM={T_em_init:.1f}_"
    f"Tnue={T_nue_init:.1f}_"
    f"Tnumu={T_numu_init:.1f}_"
    f"Tnutau={T_nutau_init:.1f}_"
    f"{suffix}.txt"
)

out_path = os.path.join(out_dir, fname)
header = "T[MeV] delta_rho_nu_total[%] delta_rho_nu_e[%] delta_rho_nu_mu[%] delta_rho_nu_tau[%]"

np.savetxt(out_path,
           np.column_stack([T_grid,
                            delta_tot_on_grid,
                            delta_e_on_grid,
                            delta_mu_on_grid,
                            delta_tau_on_grid]),
           fmt="%.6f",
           delimiter=" ",
           header=header,
           comments="")

print(f"[done] wrote {out_path} (rows={len(T_grid)})")

