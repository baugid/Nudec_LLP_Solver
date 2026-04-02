# Nudec_LLP_Solver

This is the Boltzmann solver for tracing neutrino momentum distributions to study the impact of hypothetical Long-Lived Particles (LLPs) on the cosmic neutrinos in the MeV primordial plasma.

The code has been developed by Kensuke Akita, Gideon Baur, and Maksym Ovchynnikov. The underlying physics and technical details are described in the associated preprints [2411.00892](https://arxiv.org/abs/2411.00892) and [2411.00931](https://arxiv.org/abs/2411.00931). If you use this code, please cite these references.

The code currently incorporates decaying processes of LLPs to stable electromagnetic particles (photons and electrons), metastable particles (charged pions and muons), and neutrinos.
The non-equilibrium evolution of the injected metastable particles: decays, annihilations, and interactions with nucleons can be computed in the companion code [Metastable-dynamics](https://github.com/maksymovchynnikov/Metastable-dynamics).

## How to use (key study)

Focus on EM-philic/neutrinophilic decays, and energy conservation checks. 


Key example launches:

```
python /eos/user/o/ovchynni/Nu_Decoupling_Simple/basicRunner_cli.py --llp-mass 200 --llp-lifetime 0.1 --llp-abundance 4.73 --llp-two-nu-decay-e 0.33333 --llp-two-nu-decay-mu 0.33333 --llp-two-nu-decay-tau 0.33334 --nbins 101 --ifDebugging False 
```
```
python /eos/user/o/ovchynni/Nu_Decoupling_Simple/basicRunner_cli.py --llp-mass 200 --llp-lifetime 0.1 --llp-abundance 4.73 --llp-two-nu-decay-e 0.33333 --llp-two-nu-decay-mu 0.33333 --llp-two-nu-decay-tau 0.33334 --nbins 101 --ifDebugging True
``` 

```
python /eos/user/o/ovchynni/Nu_Decoupling_Simple/basicRunner_cli.py --llp-mass 200 --llp-lifetime 0.1 --llp-abundance 4.73 --llp-two-nu-decay-e 0.33333 --llp-two-nu-decay-mu 0.33333 --llp-two-nu-decay-tau 0.33334 --nbins 201 --ifDebugging False
```
```
python /eos/user/o/ovchynni/Nu_Decoupling_Simple/basicRunner_cli.py --llp-mass 200 --llp-lifetime 0.1 --llp-abundance 4.73 --llp-two-nu-decay-e 0.33333 --llp-two-nu-decay-mu 0.33333 --llp-two-nu-decay-tau 0.33334 --nbins 201 --ifDebugging True
```

The output:

```
T_start = 5.1101422997e+00 MeV T_fin = 2.4195485397e-02 MeV n_nu_e/n_gamma = 2.1802640503e-01 n_nu_mu/n_gamma = 1.9781177313e-01 n_nu_tau/n_gamma = 1.9781173089e-01 ((a*T)_start/(a*T)_fin)^3 = 2.1973125249e-01 N_eff = 2.6724780319e+00 Accepted steps = 1124 RHS evaluations = 3647 Runtime = 326.6041 s
```

```
T_start = 5.1101422997e+00 MeV T_fin = 2.4307161185e-02 MeV n_nu_e/n_gamma = 2.1727380588e-01 n_nu_mu/n_gamma = 1.9653497346e-01 n_nu_tau/n_gamma = 1.9653493200e-01 ((a*T)_start/(a*T)_fin)^3 = 2.1671657377e-01 N_eff = 2.6481423226e+00 Accepted steps = 1071 RHS evaluations = 3338 Runtime = 404.0000 s
```

```
T_start = 5.1101422997e+00 MeV T_fin = 2.4209828181e-02 MeV n_nu_e/n_gamma = 2.0892315329e-01 n_nu_mu/n_gamma = 1.8929449964e-01 n_nu_tau/n_gamma = 1.8929445423e-01 ((a*T)_start/(a*T)_fin)^3 = 2.1934095343e-01 N_eff = 2.6669071938e+00 Accepted steps = 1143 RHS evaluations = 3803 Runtime = 2572.5953 s
```

```
T_start = 5.1101422997e+00 MeV T_fin = 2.4382902451e-02 MeV n_nu_e/n_gamma = 2.0796659183e-01 n_nu_mu/n_gamma = 1.8757746964e-01 n_nu_tau/n_gamma = 1.8757742551e-01 ((a*T)_start/(a*T)_fin)^3 = 2.1470326317e-01 N_eff = 2.6312268193e+00 Accepted steps = 1056 RHS evaluations = 3323 Runtime = 3066.2598 s
```

