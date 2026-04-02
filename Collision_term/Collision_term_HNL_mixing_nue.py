import numpy as np
from numba import jit
from Collision_term.D_function import D_function
from Constants import *
from Momentum_Grid import *

from HNL_parameters import *

@jit
def Collision_term_HNL(x,z,ni,i,f_nue,f_numu,f_nutau,delta_me,f_HNL):

    Coll = np.zeros(4)

    EN1 = (i**2 + (mN/me)**2*x**2)**(1/2) #HNL energy for Boltzmann equations for HNL


    #Collision terms      
    for nk in range(n): #bin for an integration in the collision term

        k = y_min + dy*nk

        Ek = (k**2 + x**2 + delta_me)**(1/2) #electron energy

        EN3 =   (k**2 + (mN/me)**2*x**2)**(1/2) #HNL energy with k 

        
        for nj in range(n): #bin for an integration in the collision term
            
            j = y_min + dy*nj 

            Ej = (j**2 + x**2 + delta_me)**(1/2) #electron energy
            EN2 =   (j**2 + (mN/me)**2*x**2)**(1/2) #HNL energy with j

            ############################################################################################################################## 
            ##############################################################################################################################  
            #HNL (only mixing with nue) scattering contributions 

            #N(1) + nu(2) <-> nu(3) + nu(4) for HNL

            l = EN1 + j - k
            
            if (l >= 0): 
                        
                nl_tmp = (l - y_min)/dy
                nl = round(nl_tmp)
                        
                if (nl < n):

                    l = y_min + dy*nl

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    S1_N = 2*(3*D1 - 2*D2_14/(EN1*l) - 2*D2_23/(j*k) + D2_12/(EN1*j) + D2_34/(k*l) + 3*D3/(EN1*j*k*l))
                        
                    S2_N = 2*D1 + D2_12/(EN1*j) + D2_34/(k*l) -D2_14/(EN1*l) - D2_23/(j*k) + 2*D3/(EN1*j*k*l)
                        
                    S3_N = D1 - D2_23/(j*k) - D2_14/(EN1*l) + D3/(EN1*j*k*l)

                    overall_fac = 1/2*U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*l*coe_simps[nk]*coe_simps[nj] #1/2: d.o.f. of spin of Dirac HNLs

                    Coll[3] = Coll[3] + overall_fac \
                            *((f_nue[nk]*(1 - f_HNL[ni])*f_nue[nl]*(1 - f_nue[nj]) - f_HNL[ni]*(1 - f_nue[nk])*f_nue[nj]*(1 - f_nue[nl]))*S1_N \
                            +(f_nue[nk]*(1 - f_HNL[ni])*f_numu[nl]*(1 - f_numu[nj]) - f_HNL[ni]*(1 - f_nue[nk])*f_numu[nj]*(1 - f_numu[nl]))*S2_N \
                            +(f_numu[nk]*f_numu[nl]*(1 - f_HNL[ni])*(1 - f_nue[nj]) - f_HNL[ni]*f_nue[nj]*(1-f_numu[nk])*(1 - f_numu[nl]))*S3_N \
                            +(f_nue[nk]*(1 - f_HNL[ni])*f_nutau[nl]*(1 - f_nutau[nj]) - f_HNL[ni]*(1 - f_nue[nk])*f_nutau[nj]*(1 - f_nutau[nl]))*S2_N \
                            +(f_nutau[nk]*f_nutau[nl]*(1 - f_HNL[ni])*(1 - f_nue[nj]) - f_HNL[ni]*f_nue[nj]*(1-f_nutau[nk])*(1 - f_nutau[nl]))*S3_N)

                   
            #N(4) + nu(3) <-> nu(1) + nu(2) for active nu

            EN4 = i + j - k


            if (EN4 > (mN/me)*x):


                l = (EN4**2 - (mN/me)**2*x**2)**(1/2)
                        
                nl_tmp = (l - y_min)/dy
                nl = round(nl_tmp)

                if (nl < n):
                            
                    l = y_min + dy*nl
                    EN4 = (l**2 + (mN/me)**2*x**2)**(1/2)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    S1_nu = (D1 + D2_12/(i*j) + D2_34/(k*EN4) +D3/(i*j*k*EN4))
                            
                    S2_nu = (D1 - D2_14/(i*EN4) - D2_23/(j*k) + D3/(i*j*k*EN4))
                            
                    S3_nu = 2*(D1 + D2_12/(i*j) + D2_34/(k*EN4) + D3/(i*j*k*EN4))
                            
                    S4_nu = 4*(D1 - D2_14/(i*EN4) - D2_23/(j*k) + D3/(i*j*k*EN4))
                            
                    S5_nu = (D1 - D2_14/(i*EN4) - D2_23/(j*k) + D3/(i*j*k*EN4))
                            
                    S6_nu = 4*(D1 -D2_13/(i*k) -D2_24/(j*EN4) + D3/(i*j*k*EN4)) #anti HNL contribution

                    S7_nu = (D1 -D2_13/(i*k) -D2_24/(j*EN4) + D3/(i*j*k*EN4)) #anti HNL contribution

                    overall_fac = U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*l*coe_simps[nk]*coe_simps[nj]


                    Coll[0] = Coll[0] + overall_fac \
                                            *((f_nue[nk]*f_HNL[nl]*(1-f_nue[ni])*(1-f_nue[nj]) - (1-f_nue[nk])*(1-f_HNL[nl])*f_nue[ni]*f_nue[nj])*(S3_nu + S4_nu +S6_nu) \
                                            + (f_numu[nk]*f_HNL[nl]*(1-f_nue[ni])*(1-f_numu[nj]) - (1-f_numu[nk])*(1-f_HNL[nl])*f_nue[ni]*f_numu[nj])*(S1_nu+S2_nu) \
                                            + (f_nutau[nk]*f_HNL[nl]*(1-f_nue[ni])*(1-f_nutau[nj]) - (1-f_nutau[nk])*(1-f_HNL[nl])*f_nue[ni]*f_nutau[nj])*(S1_nu + S2_nu))                        
                                
                    Coll[1] = Coll[1] + overall_fac \
                                            *((f_numu[nk]*f_HNL[nl]*(1-f_numu[ni])*(1-f_nue[nj]) - (1-f_numu[nk])*(1-f_HNL[nl])*f_numu[ni]*f_nue[nj])*S1_nu \
                                            + (f_nue[nk]*f_HNL[nl]*(1-f_numu[ni])*(1-f_numu[nj]) - (1-f_nue[nk])*(1-f_HNL[nl])*f_numu[ni]*f_numu[nj])*(S5_nu +S7_nu))
                                            
                    Coll[2] = Coll[2] + overall_fac \
                                            *((f_nutau[nk]*f_HNL[nl]*(1-f_nutau[ni])*(1-f_nue[nj]) - (1-f_nutau[nk])*(1-f_HNL[nl])*f_nutau[ni]*f_nue[nj])*S1_nu \
                                            + (f_nue[nk]*f_HNL[nl]*(1-f_nutau[ni])*(1-f_nutau[nj]) - (1-f_nue[nk])*(1-f_HNL[nl])*f_nutau[ni]*f_nutau[nj])*(S5_nu +S7_nu))
                            
                    
            #N(1) e^+-(2) <-> nu(3) e^+-(4)  for HNL

            El = EN1+Ej-k
                    
            if (El > (x**2+delta_me)**(1/2)):

                l = (El**2-x**2-delta_me)**(1/2)  

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                SS1_N = 4*(2*D1 - D2_23/(Ej*k) - D2_14/(EN1*El) + D2_34/(k*El) + D2_12/(EN1*Ej) + 2*D3/(EN1*Ej*k*El)) 

                SS2_N = 4*(x**2+delta_me)*(D1 - D2_13/(EN1*k))/(Ej*El) 


                overall_fac = 1/2*U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj] #1/2: d.o.f. of spin of Dirac HNLs

                fe_l = 1/(np.exp(El/z)+1)

                fe_j = 1/(np.exp(Ej/z)+1)


                Coll[3] = Coll[3] + 2*overall_fac \
                                        *(fe_l*(1 - fe_j)*f_nue[nk]*(1-f_HNL[ni]) \
                                        - fe_j*(1 - fe_l)*f_HNL[ni]*(1-f_nue[nk])) \
                                        *((gL**2 + gR**2)*SS1_N - gL*gR*SS2_N)
                

            #N(3) e^+-(4) <-> nu_e(1) e^+-(2)  for active nu

            El = i + Ej - EN3

            if (El > (x**2+delta_me)**(1/2)):
                        
                l = (El**2-x**2-delta_me)**(1/2)  
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)
                
                SS1_N2 = 4*(2*D1 - D2_23/(Ej*EN3) - D2_14/(i*El) + D2_34/(EN3*El) + D2_12/(i*Ej) + 2*D3/(i*Ej*EN3*El))

                SS2_N2 = 4*(x**2+delta_me)*(D1 - D2_13/(i*EN3))/(Ej*El) 
                
                overall_fac = U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj]

                fe_l = 1/(np.exp(El/z)+1)

                fe_j = 1/(np.exp(Ej/z)+1)


                Coll[0] = Coll[0] + 2*overall_fac \
                                *(fe_l*(1 - fe_j)*f_HNL[nk]*(1-f_nue[ni]) \
                                - fe_j*(1 - fe_l)*f_nue[ni]*(1-f_HNL[nk])) \
                                *((gL**2 + gR**2)*SS1_N2 - gL*gR*SS2_N2) 
                
                
            #N(1) nu(2) <-> e^-(3) e^+(4)  for HNL  

            El = EN1 + j - Ek
                    
            if (El > (x**2+delta_me)**(1/2)):
                
                l = (El**2 - x**2 - delta_me)**(1/2)
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)
                
                SSS1_N = 4*(D1 - D2_23/(j*Ek) - D2_14/(EN1*El) + D3/(EN1*j*Ek*El))

                SSS2_N = 4*(D1 -  D2_24/(j*El) - D2_13/(EN1*Ek) + D3/(EN1*j*Ek*El))

                SSS3_N = 4*(x**2+delta_me)*(D1 + D2_12/(EN1*j))/(Ek*El) 
                
                overall_fac = 1/2*U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj] #1/2: d.o.f. of spin of Dirac HNLs
                
                fe_l = 1/(np.exp(El/z)+1)

                fe_k = 1/(np.exp(Ek/z)+1)
                
                Coll[3] = Coll[3] + 2*overall_fac \
                            *(fe_l*fe_k*(1-f_HNL[ni])*(1-f_nue[nj]) \
                            - (1 - fe_j)*(1 - fe_l)*f_HNL[ni]*f_nue[nj]) \
                            *(SSS1_N*gL**2 + gR**2*SSS2_N + gL*gR*SSS3_N) 
                

            #N(2) nu(1) <-> e^-(4) e^+(3)  for active nu 

            El = i + EN2 - Ek
                    
            if (El > (x**2+delta_me)**(1/2)):
                
                l = (El**2 - x**2 - delta_me)**(1/2)
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                
                SSS1_N2 = 4*(D1 - D2_23/(EN2*Ek) - D2_14/(i*El) + D3/(i*EN2*Ek*El))

                SSS2_N2 = 4*(D1 -  D2_24/(EN2*El) - D2_13/(i*Ek) + D3/(i*EN2*Ek*El))

                SSS3_N2 = 4*(x**2+delta_me)*(D1 + D2_12/(i*EN2))/(Ek*El) 
                
                overall_fac = U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj]

                fe_l = 1/(np.exp(El/z)+1)

                fe_k = 1/(np.exp(Ek/z)+1)

                Coll[0] = Coll[0] +  2*overall_fac \
                            *(fe_l*fe_k*(1-f_HNL[ni])*(1-f_nue[nj]) \
                            -(1 - fe_k)*(1 - fe_l)*f_HNL[ni]*f_nue[nj]) \
                            *(SSS1_N2*gL**2 + gR**2*SSS2_N2 + gL*gR*SSS3_N2)  
                

            ############################################################################################################################## 
            ##############################################################################################################################  
            #HNL (only mixing with nue) decay contributions

            #HNL for N(1) <-> nu_e(2) nu_alpha(3) nu_alpha(4) #The number is the label of momentum

            l = EN1 - j - k 

            nl_tmp = (l - y_min)/dy
            nl = round(nl_tmp)

            if (l > 0) and (nl < n):

                l = y_min + dy*nl

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                overall_fac = 1/2*U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*l*coe_simps[nk]*coe_simps[nj] #1/2: D1.o.D3. of spin of Dirac HNLs    

                FunN0 = f_nue[nj]*f_nue[nk]*f_nue[nl]*(1-f_HNL[ni]) - f_HNL[ni]*(1 - f_nue[nj])*(1 - f_nue[nk])*(1- f_nue[nl]) #N<-> nue nebar nue
                FunN1 = f_nue[nj]*f_numu[nk]*f_numu[nl]*(1-f_HNL[ni]) - f_HNL[ni]*(1 - f_nue[nj])*(1 - f_numu[nk])*(1- f_numu[nl]) #N<-> nue nmubar numu
                FunN2 = f_nue[nj]*f_nutau[nk]*f_nutau[nl]*(1-f_HNL[ni]) - f_HNL[ni]*(1 - f_nue[nj])*(1 - f_nutau[nk])*(1- f_nutau[nl]) #N<-> nutau ntaubar nutau

                Coll[3] = Coll[3] + overall_fac*2*(D1 + D2_23/(j*k) -D2_14/(EN1*l) - D3/(EN1*j*k*l))*(FunN0 + (1/2)*FunN1 + (1/2)*FunN2)

            #Active neutrinos for N(2) <-> nu_e(1) nu_alpha(4) nu_alpha(3)

            EN2 = (j**2 + (mN/me)**2*x**2)**(1/2) 
            l = EN2 - i - k

            nl_tmp = (l - y_min)/dy
            nl = round(nl_tmp)

            if (l > 0) and (nl < n):

                l = y_min + dy*nl

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                overall_fac = U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*l*coe_simps[nk]*coe_simps[nj]
                
                
                Fune0 = f_HNL[nj]*(1-f_nue[ni])*(1-f_nue[nk])*(1-f_nue[nl]) - f_nue[ni]*f_nue[nk]*f_nue[nl]*(1-f_HNL[nj]) #N<-> nue nebar nue
                Fune1 = f_HNL[nj]*(1-f_nue[ni])*(1-f_numu[nk])*(1-f_numu[nl]) - f_nue[ni]*f_numu[nk]*f_numu[nl]*(1-f_HNL[nj]) #N<-> nue nmubar numu
                Fune2 = f_HNL[nj]*(1-f_nue[ni])*(1-f_nutau[nk])*(1-f_nutau[nl]) - f_nue[ni]*f_nutau[nk]*f_nutau[nl]*(1-f_HNL[nj]) #N<-> nue ntaubar nutau
                
                Funmu = f_HNL[nj]*(1-f_numu[ni])*(1-f_numu[nk])*(1-f_nue[nl]) - f_numu[ni]*f_numu[nk]*f_nue[nl]*(1-f_HNL[nj]) 
                Funtau = f_HNL[nj]*(1-f_nutau[ni])*(1-f_nutau[nk])*(1-f_nue[nl]) - f_nutau[ni]*f_nutau[nk]*f_nue[nl]*(1-f_HNL[nj]) 

                #Fun20 = f_HNL[nj]
                #Fun21 = f_HNL[nj]
                #Fun22 = f_HNL[nj]

                Coll[0] = Coll[0] + overall_fac*2*(D1 - D2_23/(EN2*k) + D2_14/(i*l) - D3/(i*EN2*k*l))*(3*Fune0 + (1/2)*Fune1 + (1/2)*Fune2)
                Coll[1] = Coll[1] + overall_fac*2*(D1 - D2_23/(EN2*k) + D2_14/(i*l) - D3/(i*EN2*k*l))*Funmu
                Coll[2] = Coll[2] + overall_fac*2*(D1 - D2_23/(EN2*k) + D2_14/(i*l) - D3/(i*EN2*k*l))*Funtau

            
            #HNL for N(1) <-> nu(2) e+(3) e-(4)
                    
            El = EN1 - j - Ek

            if (El > (x**2 + delta_me)**(1/2)):

                l = (El**2 - x**2 - delta_me)**(1/2)
                fek = 1/(np.exp(Ek/z) + 1)
                fel = 1/(np.exp(El/z) + 1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                overall_fac = 1/2*U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj] #1/2: d.o.f. of spin of Dirac HNLs 
                    
 
                Fun3 = f_nue[nj]*fek*fel*(1-f_HNL[ni])-f_HNL[ni]*(1-f_nue[nj])*(1-fek)*(1-fel)

                Coll[3] = Coll[3] + overall_fac*4*((gL**2 + gR**2)*D1 + gL**2*(D2_24/(j*El) - D2_13/(EN1*Ek)) + gR**2*(D2_23/(j*Ek) - D2_14/(EN1*El)) - (gL**2 + gR**2)*D3/(EN1*j*Ek*El) - gL*gR*(x**2 + delta_me)*(D1 - D2_12/(EN1*j))/(Ek*El))*Fun3

            #active neutrinos for N(2) <-> nu(1) e+(4) e-(3)

            Ek = (k**2 + x**2 + delta_me)**(1/2)

            El = EN2 - i - Ek

            if (El > (x**2 + delta_me)**(1/2)):

                l = (El**2 - (x**2 + delta_me))**(1/2)
                fek = 1/(np.exp(Ek/z) + 1)
                fel = 1/(np.exp(El/z) + 1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                overall_fac = U2*GF**2/(2*np.pi**3*i)*dy**2*j*k*El*coe_simps[nk]*coe_simps[nj] 

                Fun4 = f_HNL[nj]*(1-f_nue[ni])*(1-fek)*(1-fel)-f_nue[ni]*fek*fel*(1-f_HNL[nj])

                #Fun4 = f_HNL[nj]

                Coll[0] = Coll[0] + overall_fac*4*((gL**2 + gR**2)*D1 - gL**2*(D2_24/(EN2*El) - D2_13/(i*Ek)) - gR**2*(D2_23/(EN2*Ek) - D2_14/(i*El)) - (gL**2 + gR**2)*D3/(i*EN2*Ek*El) - gL*gR*(x**2 + delta_me)*(D1 - D2_12/(i*EN2))/(Ek*El))*Fun4
      
                  
                
            
    return Coll