import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

# Constants
G = 6.67e-11
c = 2.998e8
Msun = 1.99e30
Rsun = 7e8
year = 365*24*3600.

# Grid of reference eccentricities
ref_eccs = np.linspace(0.0,1.,1000)
ref_de = ref_eccs[1]-ref_eccs[0]

# Build g(e) function
g_e = np.power(ref_eccs,12./19.)*np.power(1.+(121./304.)*ref_eccs**2.,870./2299.)/(1.-ref_eccs**2.)

# Build F(e) function
ref_F_integrand = np.power(g_e,4.)*np.power(1.-ref_eccs**2.,5./2.)/(ref_eccs*(1.+(121./304.)*ref_eccs**2.))
ref_F_integrand[0] = 0
ref_F = np.cumsum(ref_F_integrand)*ref_de
ref_F *= (48./19.)/np.power(g_e,4.)

# Approximate F(e) valid at low eccentricities
ref_F_lowEcc = np.power(ref_eccs,48./19.)/np.power(g_e,4.)

class binary():
        
    def __init__(self,m1,m2,a1,a2,t1,t2,phi1,phi2,semi_a):

        self.m1 = float(m1)
        self.m2 = float(m2)
        self.a1 = a1
        self.a2 = a2
        self.s1_hat = np.array([np.sin(t1)*np.cos(phi1),np.sin(t1)*np.sin(phi1),np.cos(t1)])
        self.s2_hat = np.array([np.sin(t2)*np.cos(phi2),np.sin(t2)*np.sin(phi2),np.cos(t2)])
        self.semi_a = float(semi_a)
        self.initial_semi_a = float(semi_a)
        self.L_hat = np.array([0,0,1])
        self.eccentricity = 0.

    def v_orb(self):
        return np.sqrt(G*(self.m1+self.m2)*Msun/(self.semi_a))

    def time_to_merger_circ(self):
        Mtot = (self.m1+self.m2)*Msun
        mu = (self.m1*self.m2/(self.m1+self.m2))*Msun
        return (5./256.)*(c**5*self.semi_a**4/(G**3.*Mtot**2.*mu))

    def time_to_merger(self):
        t0 = self.time_to_merger_circ()
        if self.eccentricity==0:
            F = 1.
        elif self.eccentricity>=1:
            return np.inf
        elif self.eccentricity>0.99:
            F = (768./429.)*(1.-self.eccentricity)**(3.5)
        elif self.eccentricity<=0.075:
            F = np.interp(self.eccentricity,ref_eccs,ref_F_lowEcc)
        else:
            F = np.interp(self.eccentricity,ref_eccs,ref_F)
        return t0*F

    def period(self):
        return 2.*np.pi*self.semi_a**1.5/np.sqrt(G*(self.m1+self.m2)*Msun)

    def chi_effective(self):
        cost1 = self.s1_hat.dot(self.L_hat)
        cost2 = self.s2_hat.dot(self.L_hat)
        return (self.m1*self.a1*cost1+self.m2*self.a2*cost2)/(self.m1+self.m2)

    def chi_p(self):

        cost1 = self.s1_hat.dot(self.L_hat)
        cost2 = self.s2_hat.dot(self.L_hat)

        # Mass ratio 
        q_inv = 1.0*self.m1/self.m2

        # Coefficients A1 and A2 
        A1 = 2 + 3/(2*q_inv)
        A2 = 2 + 3*q_inv/2

        # Calculating sin from cos
        sintilt1 = np.sqrt(1-(cost1)**2)
        sintilt2 = np.sqrt(1-(cost2)**2)

        # Terms in Xp
        term1 = A1*self.a1*(self.m1**2)*sintilt1
        term2 = A2*self.a2*(self.m2**2)*sintilt2
        coeff = 1.0/(2*A1*(self.m1**2))

        # Using addition of terms method 
        Xp = coeff*(term1 + term2 + np.abs(term1 - term2))
        return Xp
        
    def bindingEnergy(self):
        m1_SI = self.m1*Msun
        m2_SI = self.m2*Msun
        return -G*m1_SI*m2_SI/(2.*self.semi_a)

    def kick(self,vk,beta):

        self.m2 = beta*self.m2

        mu = (self.m1*self.m2)/(self.m1+self.m2)*Msun
        M = (self.m1+self.m2)*Msun
        vk_SI = np.array(vk)*1e3
        v_rel = self.v_orb()

        # Precompute various things
        vkx_SI_2 = vk_SI[0]*vk_SI[0]
        vky_SI_2 = vk_SI[1]*vk_SI[1]
        vkz_SI_2 = vk_SI[2]*vk_SI[2]
        vk_SI_2 = vkx_SI_2 + vky_SI_2 + vkz_SI_2

        vk_dot_vorb = vk_SI[1]*v_rel
        v_rel_2 = v_rel*v_rel

        a_post = G*M/((2.*G*M)/self.semi_a - vk_SI_2 - v_rel_2 - 2.*vk_dot_vorb)
        e_post = np.sqrt(1.-(vkz_SI_2 + vky_SI_2 + v_rel_2 + 2.*vk_dot_vorb)*self.semi_a*self.semi_a/(G*M*a_post))
        self.L_hat = np.array([0.,-vk_SI[2],v_rel+vk_SI[1]])/(vky_SI_2 + v_rel_2 + 2.*vk_dot_vorb + vkz_SI_2)**0.5

        """
        r_hat = np.array([1,0,0])       
        vk_SI = np.array(vk)*1e3        
        vorb_SI = np.array([0.,1.,0.])*self.v_orb()

        # Post kick angular momentum
        L = mu*np.cross(self.semi_a*r_hat,vorb_SI+vk_SI)
        L_mag = np.sqrt(L.dot(L))
        L_final_hat = L/L_mag
        self.L_hat = L_final_hat

        # Post kick energy
        E_kin = 0.5*mu*(vk_SI+vorb_SI).dot(vk_SI+vorb_SI)
        E_pot = -G*M*mu/self.semi_a
        E_tot = E_kin+E_pot

        ecc = np.sqrt(1.+(2.*E_tot*L_mag**2.)/(mu*(G*M*mu)**2.))      
        self.eccentricity = ecc
        self.semi_a = -(G*M*mu)/(2.*E_tot)
        """

        self.eccentricity = e_post
        self.semi_a = a_post
        
        F = 2.*beta - 1. - vk_SI_2/v_rel_2 -2.*vk_dot_vorb/v_rel_2
        if F<0:
            return 0
        else:
            return 1
