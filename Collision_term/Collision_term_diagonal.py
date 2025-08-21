"""
This file implements the collision term for momentum distribution functions (not for the density matrix). In general, it is able to handle arbitrary grids.
See Section 2.2.4, Appendix A and B in K. Akita and M. Yamaguchi, arXiv: 2210.10307 for details.
"""

import numpy as np
from numba import jit
from Collision_term.D_function import D_function
from Constants import *
import Momentum_Grid


@jit(nopython=True, nogil=True, fastmath=True)
def Collision_term_diagonal(x, z, ni, i, f_nue, f_numu, f_nutau, f_nue_bar, f_numu_bar, f_nutau_bar, delta_me):
    Coll = np.zeros(3)

    # Collision terms
    for nk in range(Momentum_Grid.n):  # bin for an integration in the collision term

        k = Momentum_Grid.gridVals[nk]

        for nj in range(Momentum_Grid.n):  # bin for an integration in the collision term

            j = Momentum_Grid.gridVals[nj]

            # neutrino self-interations
            # We do not distinguish neutrinos and anti-neutrinos in neutrino-self interactions
            lPre = i + j - k
            if Momentum_Grid.y_min <= lPre <= Momentum_Grid.y_max:

                # Choose absolute closest grid point
                nlPre = np.searchsorted(Momentum_Grid.gridVals, lPre)
                if nlPre == 0:
                    nl = 0
                elif nlPre == Momentum_Grid.n:
                    nl = nlPre - 1
                elif Momentum_Grid.gridVals[nlPre] - lPre > lPre - Momentum_Grid.gridVals[nlPre - 1]:
                    nl = nlPre - 1
                else:
                    nl = nlPre
                # For non-linear grids this would differ from lPre and violate momentum conservation.
                # Conversely, keeping lPre would break it down the line
                l = Momentum_Grid.gridVals[nl]

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i, j, k, l)

                Pi_self1 = 2 * (3 * D1 - 2 * D2_14 / (i * l) - 2 * D2_23 / (j * k) + D2_12 / (i * j) + D2_34 / (
                        k * l) + 3 * D3 / (i * j * k * l))

                Pi_self2 = 2 * D1 + D2_12 / (i * j) + D2_34 / (k * l) - D2_14 / (i * l) - D2_23 / (j * k) + 2 * D3 / (
                        i * j * k * l)

                Pi_self3 = D1 - D2_23 / (j * k) - D2_14 / (i * l) + D3 / (i * j * k * l)

                overall_fac_self = GF ** 2 / (2 * np.pi ** 3 * i) * j * k * l * Momentum_Grid.gridWeights[nk] * \
                                   Momentum_Grid.gridWeights[nj]

                # eq. (62) in 2005.07047

                Coll[0] = Coll[0] + overall_fac_self \
                          * ((f_nue[nk] * (1 - f_nue[ni]) * f_nue[nl] * (1 - f_nue[nj]) - f_nue[ni] * (1 - f_nue[nk]) *
                              f_nue[nj] * (1 - f_nue[nl])) * Pi_self1 \
                             + (f_nue[nk] * (1 - f_nue[ni]) * f_numu[nl] * (1 - f_numu[nj]) - f_nue[ni] * (
                                1 - f_nue[nk]) * f_numu[nj] * (1 - f_numu[nl])) * Pi_self2 \
                             + (f_numu[nk] * f_numu[nl] * (1 - f_nue[ni]) * (1 - f_nue[nj]) - f_nue[ni] * f_nue[nj] * (
                                1 - f_numu[nk]) * (1 - f_numu[nl])) * Pi_self3 \
                             + (f_nue[nk] * (1 - f_nue[ni]) * f_nutau[nl] * (1 - f_nutau[nj]) - f_nue[ni] * (
                                1 - f_nue[nk]) * f_nutau[nj] * (1 - f_nutau[nl])) * Pi_self2 \
                             + (f_nutau[nk] * f_nutau[nl] * (1 - f_nue[ni]) * (1 - f_nue[nj]) - f_nue[ni] * f_nue[
                            nj] * (1 - f_nutau[nk]) * (1 - f_nutau[nl])) * Pi_self3)

                Coll[1] = Coll[1] + overall_fac_self \
                          * ((f_numu[nk] * (1 - f_numu[ni]) * f_numu[nl] * (1 - f_numu[nj]) - f_numu[ni] * (
                        1 - f_numu[nk]) * f_numu[nj] * (1 - f_numu[nl])) * Pi_self1 \
                             + (f_numu[nk] * (1 - f_numu[ni]) * f_nue[nl] * (1 - f_nue[nj]) - f_numu[ni] * (
                                1 - f_numu[nk]) * f_nue[nj] * (1 - f_nue[nl])) * Pi_self2 \
                             + (f_nue[nk] * f_nue[nl] * (1 - f_numu[ni]) * (1 - f_numu[nj]) - f_numu[ni] * f_numu[
                            nj] * (1 - f_nue[nk]) * (1 - f_nue[nl])) * Pi_self3 \
                             + (f_numu[nk] * (1 - f_numu[ni]) * f_nutau[nl] * (1 - f_nutau[nj]) - f_numu[ni] * (
                                1 - f_numu[nk]) * f_nutau[nj] * (1 - f_nutau[nl])) * Pi_self2 \
                             + (f_nutau[nk] * f_nutau[nl] * (1 - f_numu[ni]) * (1 - f_numu[nj]) - f_numu[ni] * f_numu[
                            nj] * (1 - f_nutau[nk]) * (1 - f_nutau[nl])) * Pi_self3)

                Coll[2] = Coll[2] + overall_fac_self \
                          * ((f_nutau[nk] * (1 - f_nutau[ni]) * f_nutau[nl] * (1 - f_nutau[nj]) - f_nutau[ni] * (
                        1 - f_nutau[nk]) * f_nutau[nj] * (1 - f_nutau[nl])) * Pi_self1 \
                             + (f_nutau[nk] * (1 - f_nutau[ni]) * f_nue[nl] * (1 - f_nue[nj]) - f_nutau[ni] * (
                                1 - f_nutau[nk]) * f_nue[nj] * (1 - f_nue[nl])) * Pi_self2 \
                             + (f_nue[nk] * f_nue[nl] * (1 - f_nutau[ni]) * (1 - f_nutau[nj]) - f_nutau[ni] * f_nutau[
                            nj] * (1 - f_nue[nk]) * (1 - f_nue[nl])) * Pi_self3 \
                             + (f_nutau[nk] * (1 - f_nutau[ni]) * f_numu[nl] * (1 - f_numu[nj]) - f_nutau[ni] * (
                                1 - f_nutau[nk]) * f_numu[nj] * (1 - f_numu[nl])) * Pi_self2 \
                             + (f_numu[nk] * f_numu[nl] * (1 - f_nutau[ni]) * (1 - f_nutau[nj]) - f_nutau[ni] * f_nutau[
                            nj] * (1 - f_numu[nk]) * (1 - f_numu[nl])) * Pi_self3)

            # nu e^+- <-> nu e^+-

            E = (j ** 2 + x ** 2 + delta_me) ** (1 / 2)
            El = i + E - k

            if (El > (x ** 2 + delta_me) ** (1 / 2)):
                l = (El ** 2 - x ** 2 - delta_me) ** (1 / 2)

                fe_j = 1 / (np.exp(E / z) + 1)
                fe_l = 1 / (np.exp(El / z) + 1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i, j, k, l)

                Pi_sc1 = 2 * (2 * D1 - D2_23 / (E * k) - D2_14 / (i * El) + D2_34 / (k * El) + D2_12 / (
                        i * E) + 2 * D3 / (i * E * k * El))

                Pi_sc2 = 2 * (x ** 2 + delta_me) * (D1 - D2_13 / (i * k)) / (E * El)

                overall_fac_sc = GF ** 2 / (2 * np.pi ** 3 * i) * Momentum_Grid.gridWeights[nk] * \
                                 Momentum_Grid.gridWeights[nj] * j * k * El

                Coll[0] = Coll[0] + overall_fac_sc \
                          * (fe_l * (1 - fe_j) * f_nue[nk] * (1 - f_nue[ni]) * (
                        Pi_sc1 * 2 * (gL ** 2 + gR ** 2) - Pi_sc2 * 4 * gL * gR) \
                             - fe_j * (1 - fe_l) * f_nue[ni] * (1 - f_nue[nk]) * (
                                     Pi_sc1 * 2 * (gL ** 2 + gR ** 2) - Pi_sc2 * 4 * gL * gR))

                Coll[1] = Coll[1] + overall_fac_sc \
                          * (fe_l * (1 - fe_j) * f_numu[nk] * (1 - f_numu[ni]) * (
                        Pi_sc1 * 2 * (gLtilde ** 2 + gR ** 2) - Pi_sc2 * 4 * gLtilde * gR) \
                             - fe_j * (1 - fe_l) * f_numu[ni] * (1 - f_numu[nk]) * (
                                     Pi_sc1 * 2 * (gLtilde ** 2 + gR ** 2) - Pi_sc2 * 4 * gLtilde * gR))

                Coll[2] = Coll[2] + overall_fac_sc \
                          * (fe_l * (1 - fe_j) * f_nutau[nk] * (1 - f_nutau[ni]) * (
                        Pi_sc1 * 2 * (gLtilde ** 2 + gR ** 2) - Pi_sc2 * 4 * gLtilde * gR) \
                             - fe_j * (1 - fe_l) * f_nutau[ni] * (1 - f_nutau[nk]) * (
                                     Pi_sc1 * 2 * (gLtilde ** 2 + gR ** 2) - Pi_sc2 * 4 * gLtilde * gR))

            # nu nu <-> e^- e^+

            E = (k ** 2 + x ** 2 + delta_me) ** (1 / 2)
            El = i + j - E

            if (El > (x ** 2 + delta_me) ** (1 / 2)):
                l = (El ** 2 - x ** 2 - delta_me) ** (1 / 2)

                fe_k = 1 / (np.exp(E / z) + 1)
                fe_l = 1 / (np.exp(El / z) + 1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i, j, k, l)

                Pi_ann1 = 2 * (D1 - D2_23 / (j * E) - D2_14 / (i * El) + D3 / (i * j * E * El))

                Pi_ann2 = 2 * (D1 - D2_24 / (j * El) - D2_13 / (i * E) + D3 / (i * j * E * El))

                Pi_ann3 = (x ** 2 + delta_me) * (D1 + D2_12 / (i * j)) / (E * El)

                overall_fac_ann = GF ** 2 / (2 * np.pi ** 3 * i) * Momentum_Grid.gridWeights[nk] * \
                                  Momentum_Grid.gridWeights[nj] * j * k * El

                Coll[0] = Coll[0] + overall_fac_ann \
                          * (fe_l * fe_k * (1 - f_nue[ni]) * (1 - f_nue_bar[nj]) * (
                        Pi_ann1 * 2 * gL ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gL * gR) \
                             - (1 - fe_k) * (1 - fe_l) * f_nue[ni] * f_nue_bar[nj] * (
                                     Pi_ann1 * 2 * gL ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gL * gR))

                Coll[1] = Coll[1] + overall_fac_ann \
                          * (fe_l * fe_k * (1 - f_numu[ni]) * (1 - f_numu_bar[nj]) * (
                        Pi_ann1 * 2 * gLtilde ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gLtilde * gR) \
                             - (1 - fe_k) * (1 - fe_l) * f_numu[ni] * f_numu_bar[nj] * (
                                     Pi_ann1 * 2 * gLtilde ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gLtilde * gR))

                Coll[2] = Coll[2] + overall_fac_ann \
                          * (fe_l * fe_k * (1 - f_nutau[ni]) * (1 - f_nutau_bar[nj]) * (
                        Pi_ann1 * 2 * gLtilde ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gLtilde * gR) \
                             - (1 - fe_k) * (1 - fe_l) * f_nutau[ni] * f_nutau_bar[nj] * (
                                     Pi_ann1 * 2 * gLtilde ** 2 + Pi_ann2 * 2 * gR ** 2 + Pi_ann3 * 4 * gLtilde * gR))

    return Coll
