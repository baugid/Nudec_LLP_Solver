# Nudec_LLP_Solver

This is the Boltzmann solver for tracing neutrino momentum distributions to study the impact of hypothetical Long-Lived Particles (LLPs) on the cosmic neutrinos in the MeV primordial plasma.

The code has been developed by Kensuke Akita, Gideon Baur, and Maksym Ovchynnikov. The underlying physics and technical details are described in the associated preprints [2411.00892](https://arxiv.org/abs/2411.00892) and [2411.00931](https://arxiv.org/abs/2411.00931). If you use this code, please cite these references.

The code currently incorporates decaying processes of LLPs to stable electromagnetic particles (photons and electrons), metastable particles (charged pions and muons), and neutrinos.
The non-equilibrium evolution of the injected metastable particles: decays, annihilations, and interactions with nucleons can be computed in the companion code [Metastable-dynamics](https://github.com/maksymovchynnikov/Metastable-dynamics).

## How to use

Focus on EM-philic/neutrinophilic decays. Key example launches:

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