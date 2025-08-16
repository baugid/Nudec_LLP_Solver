# Nudec_LLP_Solver

This is the Boltzmann solver for tracing neutrino momentum distributions to study the impact of hypothetical Long-Lived Particles (LLPs) on the cosmic neutrinos in the MeV primordial plasma.

The code has been developed by Kensuke Akita, Gideon Baur, and Maksym Ovchynnikov. The underlying physics and technical details are described in the associated preprints [2411.00892](https://arxiv.org/abs/2411.00892) and [2411.00931](https://arxiv.org/abs/2411.00931). If you use this code, please cite these references.

The code currently incorporates decaying processes of LLPs to stable electromagnetic particles (photons and electrons), metastable particles (charged pions and muons), and neutrinos.
The non-equilibrium evolution of the injected metastable particles: decays, annihilations, and interactions with nucleons can be computed in the companion code [Metastable-dynamics](https://github.com/maksymovchynnikov/Metastable-dynamics).

## The code structure

The code contains the following main scripts in Python:

SingleShot.py: is the starting script for a fixed parameter of LLP. 

BatchLauncher.py: is the starting script for the search of the parameter space of LLPs.

Core.py: is the core script that instructs the simulations and exports the results.

System_Nudecoupling.py: describes the closed set of the evolution equations for neutrinos, electromagnetic plasma temperature, and LLPs. 
The main body is based on the work by K. Akita and M. Yamaguchi: [2005.07047](https://arxiv.org/abs/2005.07047) and [2210.10307](https://arxiv.org/abs/2210.10307). 
Details of the application to LLPs are provided in Section V and Appendix C of [2411.00931](https://arxiv.org/abs/2411.00931).

Distributions.py: describes neutrino momentum distributions produced by one metastable particle decay. 

## How to use

See [HowToUse.md]().
