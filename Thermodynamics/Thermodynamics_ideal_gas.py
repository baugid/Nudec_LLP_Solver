import numpy as np
from scipy import integrate
from Constants import *
import Momentum_Grid


def qed_y_grid(z):
    """QED/electron grid with a fixed dimensionless u = y/z range."""
    return z * np.linspace(Momentum_Grid.yQED_min, Momentum_Grid.yQED_max, Momentum_Grid.n_QED)


def Energy_density_ideal_gas(x,z,f_nue,f_numu,f_nutau,f_nue_bar,f_numu_bar,f_nutau_bar):

    #energy densities in comoving volume

    y_e = qed_y_grid(z) #(comoving) momentum

    Integrand_rho_e_bar = y_e**2*(y_e**2 + x**2)**(1/2)/(np.exp((y_e**2 + x**2)**(1/2)/z) + 1)

    y = Momentum_Grid.gridVals
    Integrand_rho_nu_bar = y**3*(f_nue+f_numu+f_nutau+f_nue_bar+f_numu_bar+f_nutau_bar)

    rho_e_bar = 2/(np.pi**2)*integrate.simpson(Integrand_rho_e_bar, x=y_e) #Energy density for e^\pm in comoving volume
    rho_nu_bar = 1/(2*np.pi**2)*np.sum(Integrand_rho_nu_bar*Momentum_Grid.gridWeights) #Total neutrino and anti-neutrino energy density

    return rho_e_bar, rho_nu_bar


def Energy_density_ideal_gas_mu_pi(x,z):

    #energy densities in comoving volume

    y = Momentum_Grid.gridVals #(comoving) momentum

    Integrand_rho_mu_bar = y**2*(y**2 + (mmu/me)**2*x**2)**(1/2)/(np.exp((y**2 + (mmu/me)**2*x**2)**(1/2)/z) + 1)
    rho_mu_bar = 2/(np.pi**2)*np.sum(Integrand_rho_mu_bar*Momentum_Grid.gridWeights) #Energy density for mu^\pm in comoving volume


    Integrand_rho_pi_bar = 3/(2*np.pi**2)*y**2*(y**2 + (mpi/me)**2*x**2)**(1/2)/(np.exp((y**2 + (mpi/me)**2*x**2)**(1/2)/z) - 1)
    rho_pi_bar = np.sum(Integrand_rho_pi_bar*Momentum_Grid.gridWeights) #energy density for total pion in comoving volume

    return rho_mu_bar, rho_pi_bar


def Functions_in_z_ideal_gas(x,z):

    J_return = J(x,z)
    Y_return = Y(x,z)

    return J_return, Y_return

def Functions_in_z_ideal_gas_mu_pi(x,z):


    Jmu_return = Jmu(x,z)

    Ymu_return = Ymu(x,z)

    Jpi_return = Jpi(x,z)

    Ypi_return = Ypi(x,z)


    return Jmu_return, Ymu_return, Jpi_return, Ypi_return


def J(x,z):

    y    = qed_y_grid(z)
    u    = y/z
    w    = x/z
    dudy = 1/z

    Integrand_J = u**2*np.exp((u**2 + w**2)**(1/2))/(np.exp((u**2+w**2)**(1/2)) + 1)**2

    J = dudy*1/(np.pi**2)*integrate.simpson(Integrand_J,x=y)

    return J



def Y(x,z):

    y    = qed_y_grid(z)
    u    = y/z
    w    = x/z
    dudy = 1/z

    Integrand_Y = u**4*np.exp((u**2 + w**2)**(1/2))/(np.exp((u**2+w**2)**(1/2)) + 1)**2

    Y =  dudy*1/(np.pi**2)*integrate.simpson(Integrand_Y,x=y)

    return Y

def Jmu(x,z):

    y    = Momentum_Grid.gridVals
    u    = y/z
    w    = x/z
    dudi = 1/z

    Integrand_Jmu = dudi*1/(np.pi**2)*u**2*np.exp((u**2 + (mmu/me)**2*w**2)**(1/2))/(np.exp((u**2+(mmu/me)**2*w**2)**(1/2)) + 1)**2

    Jmu = integrate.simpson(Integrand_Jmu,x=y)

    return Jmu


def Jpi(x,z):

    y    = Momentum_Grid.gridVals
    u    = y/z
    w    = x/z
    dudi = 1/z

    Integrand_Jpi = dudi*1/(np.pi**2)*u**2*np.exp((u**2 + (mpi/me)**2*w**2)**(1/2))/(np.exp((u**2+(mpi/me)**2*w**2)**(1/2)) - 1)**2

    Jpi = np.sum(Integrand_Jpi*Momentum_Grid.gridWeights)

    return Jpi


def Ymu(x,z):

    y    = Momentum_Grid.gridVals
    u    = y/z
    w    = x/z
    dudi = 1/z

    Integrand_Ymu = dudi*1/(np.pi**2)*u**4*np.exp((u**2 + (mmu/me)**2*w**2)**(1/2))/(np.exp((u**2 + (mmu/me)**2*w**2)**(1/2)) + 1)**2

    Ymu =  np.sum(Integrand_Ymu*Momentum_Grid.gridWeights)

    return Ymu


def Ypi(x,z):

    y    = Momentum_Grid.gridVals
    u    = y/z
    w    = x/z
    dudi = 1/z

    Integrand_Ypi = dudi*1/(np.pi**2)*u**4*np.exp((u**2 + (mpi/me)**2*w**2)**(1/2))/(np.exp((u**2 + (mpi/me)**2*w**2)**(1/2)) - 1)**2

    Ypi =  np.sum(Integrand_Ypi*Momentum_Grid.gridWeights)

    return Ypi
