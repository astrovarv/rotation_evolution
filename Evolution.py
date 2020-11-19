#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:41:56 2020

@author: Varvara Semenova

"""
import numpy as np
import mpmath as mp
import time
import pandas as pd
from mpmath import lambertw
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from mpmath.calculus.optimization import Illinois
import os

# PHYSICAL CONSTANTS USED
M_sun = 1.99e33 # Solar mass
R_sun = 6.96e10 # Solar radius
G = 6.673e-8 # Gravitational constant
k = 1.38e-16 # Boltzmann constant
T = 1e6 # Assume constant coronal temperature of 10^6 K
m_H = 1.673e-24 # Hydrogen mass
R_J = 7.14e9 # Jupyter radius
cs = np.sqrt(3*k*T/m_H) # Sound speed
number_density = 1e10 # number density at the coronal base

# SET ARBITRARY PRECISION
mp.mp.dps = 50 


def find_nearest(array, value):
    """
    Finds the closest value in an array to the given one.
    
    Returns:
    -------
    idx : INDEX OF THE CLOSEST VALUE
    
    array[idx] : THE CLOSEST VALUE
        
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx, array[idx]


class Star:
    """
    
    """
    
    def __init__(self, M = None, R = None, P = None, B = None):
        
        self.__mass = M
        self.__radius = R
        self.__period = P
        if self.__period != None:
            self.__omega = 2*np.pi/(P*24*60*60)
        else: self.__omega = None
        self.__B = B
        
        # Ratio of rotational to gravitational energy:
        self.kappa = mp.mpf((self.__omega**2)*(self.__radius**3)/(G*self.__mass))
        # Ratio of gravitational to thermal energy:
        self.l = mp.mpf(G*self.__mass/(self.__radius*cs**2))
        # Ratio of magnetic to thermal energy:
        self.zeta = mp.mpf(self.__B**2/(8*np.pi*number_density*m_H*cs**2))
        self.last_fieldline = self.last_fieldline()
        
    def set_mass(self, M):
        self.__mass = M
    
    def get_mass(self):
        print(self.__mass)
    
    def set_radius(self, R):
        self.__radius = R
        
    def get_radius(self):
        print(self.__radius)
        
    def set_period(self, P):
        self.__period = P                   
        self.__omega = 2*np.pi/(self.__period*24*60*60)
        
    def get_period(self):
        print(self.__period)
        
    def get_omega(self):
        print(self.__omega)
        
    def get_Jtotal(self):
        print('Angular Mom. = ', 2/5*self.__mass*self.__radius**2*self.__omega, \
              'gcm^2/s')
        
    def dipolar_field(self, r, S):
        """
        The magnetic field strength distance r away from the star at 
        colatitude S.
        Calculated from the dipolar field equation.
        Mestel 1968 Eq. (12)
        """
        return mp.sqrt(mp.power(self.__B,2))/(mp.power((r/self.__radius),6)*(1 - 0.75*S))

    
    def __x_a(self, S):
        """
        Calculates sonic radius x_a for latitude S = sin^2(theta).
        Mestel 1968 Eq. (60)
        """

        # Solve the polynomial
        coef = [-9*self.kappa*self.l*(S**2), 12*self.l*self.kappa*S, 0, \
                -15*S, 6*S*self.l+24, -8*self.l]
        roots = np.roots(coef)
        
        # Only take the smallest real positive solution
        physical_roots = []
        for i in roots:
            if np.imag(i) == 0 and i > 0:
                physical_roots.append(np.real(i))
                
        x_a = mp.mpf(min(physical_roots))
        
        return x_a

    def __const_A(self, S):
        """
        Calculates integration constant A at latitude S = sin^2(theta), using 
        the nozzle-point x_a. 
        Mestel 1968 Eq. (61)
        """
        # Calculate the sonic radius first
        x = self.__x_a(S)
        
        A = 0.5*(1+mp.log(2)) - (self.l/x + 3*mp.log(x) \
                                 - 0.5*mp.log(1-0.75*x*S) \
                                 + 0.5*self.kappa*self.l*S*mp.power(x,3))
        return A

    def __base_velocity(self, S):
        """
        Calculates poloidal wind velocity at the coronal base at given 
        S = sin^2(theta) using the Lambert W function.
        Mestel 1968 Eq. (62)
        
        Returns
        -------
        x : SONIC RADIUS (x_a).
        
        Us : DIMENSIONLESS WIND VELOCITY AT THE CORONAL BASE.
        
        v_p : WIND VELOCITY AT THE CORONAL BASE (CGS UNITS).
        """
        
        x = self.__x_a(S)
        B = self.l*(1+self.kappa*S/2) - 0.5*mp.log(1-0.75*S)
        
        if x < 1:
            Us = 1/mp.sqrt(2)
            v_p = cs
            
        else:
            Us_squared = mp.re(-0.5*lambertw(-2*mp.exp(-2*(self.__const_A(S) + B)), 0))
            Us = mp.sqrt(Us_squared)
            v_p = mp.sqrt(2*mp.power(cs,2)*Us_squared)
            
        return x, Us, v_p
    
    
    def __last_fieldline_brent(self, x):
        """
        The last open fieldline equation put in the formed used by the brentq 
        solver. Needed to find initial "guess" parameter.
        Owen, Adams 2014 Eq.(39)
        """

        C = mp.log(self.zeta)
        
        return self.l*(x-1) + 0.5*self.kappa*self.l*(mp.power(x,-2) - x) - C - 6*mp.log(x)
    
    
    def last_fieldline(self):
        """
        Finds the last open fieldline q_m = sin^2(theta_m). 
        Owen, Adams 2014 Eq. (39)
        """
        
        def eq_to_solve(x):
            """
            The equation for q_m put in the form that is used to find roots.
            """
            C = mp.log(self.zeta)
            
            return self.l*(x-1) + 0.5*self.kappa*self.l*(mp.power(x,-2) - x) - C - 6*mp.log(x)
        
        # intial guess parameter
        root1 = brentq(self.__last_fieldline_brent, 1e-9, 1.0)
        print(root1)
        # find the last open fieldline
        x0 = mp.findroot(eq_to_solve, (root1-0.01*root1, root1+0.01*root1), solver = Illinois)
        print(x0)
        return x0
    
    def __poly_xc(self, S):
        """
        This uses the simple roots finder to find all roots of the polynomial 
        equation for the critical radius x_c. It then chooses the smallest real positive root. Used as
        an intial "guess" parameter.
        Overleaf 2020 Eq. () 
        Mestel 1968 Eq. (63) + Eq. (64)
        """
        Us = self.__base_velocity(S)[1]
        
        k9 = 0.5*mp.power(Us,2)*self.kappa*self.l*S
        k6 = mp.power(Us,2)*mp.log(mp.sqrt(1-0.75*S)*self.zeta/Us) + \
               mp.power(Us,2)*self.__const_A(S)
        k5 = mp.power(Us,2)*self.l
        k1 = 0.75*S*mp.power(self.zeta,2) - \
                9/16*mp.power(S,2)*mp.power(self.zeta,2)
        k0 = 0.75*S*mp.power(self.zeta,2) - mp.power(self.zeta,2)
        
        coef = [k9, 0, 0, k6, k5, 0, 0, 0, k1, k0]
        roots = np.roots(coef)
        
        physical_roots = []
        for i in roots:
            if np.imag(i) == 0 and i > 0:
                physical_roots.append(np.real(i))
                
        x_c = min(physical_roots)
        
        return x_c
    
    def __crit_velocity(self, S):
        """
        Finds the critical radius x_c and the critical (Alfven) velocity U_c.
        Overleaf 2020 Eq. () 
        Mestel 1968 Eq. (63) + Eq. (64)
        """
        
        Us = self.__base_velocity(S)[1]

        def xc_equation(x):
            """
            This gives the LHS of the polynomial equation for x_c to be used by the 
            Brent's root finder method.
            """
            k9 = 0.5*mp.power(Us,2)*self.kappa*self.l*S
            k6 = mp.power(Us,2)*mp.log(mp.sqrt(1-0.75*S)*self.zeta/Us) + \
                   mp.power(Us,2)*self.__const_A(S)
            k5 = mp.power(Us,2)*self.l
            k1 = 0.75*S*mp.power(self.zeta,2) - \
                    9/16*mp.power(S,2)*mp.power(self.zeta,2)
            k0 = 0.75*S*mp.power(self.zeta,2) - mp.power(self.zeta,2)
            
            return k9*mp.power(x,9) + k6*mp.power(x,6) + k5*mp.power(x,5) + k1*x + k0
        
        # initial guess parameter
        root1 = self.__poly_xc(S)
        
        # find x_c
        x_c = mp.findroot(xc_equation, (root1-0.01*root1, root1+0.01*root1), Illinois)
        
        # calculate the critical velocity from the values of x_c
        # Mestel Eq. (64)
        U_c = mp.sqrt(1 - 0.75*x_c*S)*mp.sqrt(1 - 0.75*S)*self.zeta/(Us*mp.power(x_c,3))
        
        return x_c, U_c
    
    def calculate_xc(self, S_num):
        """
        Calculate the critical radius and critical velocity for a given star for a 
        range of S values from the pole up to the last open fieldline. The S values
        are sin(theta)-spaced in this range. There are a total of S_num fieldlines 
        being calculated.
        """
    
        x_c = []
        U_c = []
        
        S1 = mp.linspace(mp.power(10,-2.25), mp.sqrt(self.last_fieldline), S_num) #sin theta spaced
        S = []
        for i in range(len(S1)):
            S.append(mp.power(S1[i],2))
        
        for i in range(len(S)):
            xc, Uc = self.__crit_velocity(S[i])
            x_c.append(xc)
            U_c.append(Uc)
        
        return S, x_c, U_c
    
    def plot_Alfven(self, S_num, filepath = None):
        
        """
        Make plots of Alfven surface and Alfven wind velocity as functions of
        latitude S.
        Adding filepath will save the plots.

        """
        S, x_c, U_c = self.calculate_xc(S_num)
        
        x_c = [float(mp.re(i)) for i in x_c]
        U_c = [float(mp.re(i)) for i in U_c]
        S = [float(mp.re(i)) for i in S]
        
        fig, ax = plt.subplots(2,1, sharex = True)
        
        ax[0].semilogy(S, x_c, '.-', label = '$x_c$ solutions')
        ax[0].axvline(x = self.last_fieldline, color = 'k', label = r'$S_m$ (last open field line)')
        ax[0].set_ylabel(r'$x_c$')
        ax[0].legend()
        ax[0].set_title(r'$log(\kappa) = {}$,'.format(np.round(np.log10(float(self.kappa)), decimals = 2)) + \
            r' $L = {}$,'.format(np.round(float(self.l), decimals = 2)) + \
        r' $\zeta = {}$'.format(np.round(float(self.zeta), decimals = 2)) + '\n Alfven Surface')
        
        ax[1].semilogy(S, U_c, '.-', label = '$U_c$ solutions', color = 'tab:red')
        ax[1].axvline(self.last_fieldline, color = 'k', \
                    label = r'$S_m$ (last open field line)')
        ax[1].set_xlabel(r'$S$')
        ax[1].set_ylabel(r'$U_c$')
        ax[1].set_title('Alfven Velocity')
        ax[1].legend()
        plt.tight_layout()
            
        if filepath:
            plt.savefig(filepath, dpi = 200)
        
        
    def crit_field(self, S_num):
        """
        Calculate the magnetic field and density at Alfven surface for a
        given number of S (latitide values).
        """
        S, x_c, U_c = self.calculate_xc(S_num)
    
        v_A = []
        B_c = []
        rho_c = []
        
        for i in range(len(S)):
            v = U_c[i]*cs*np.sqrt(2)
            B = np.sqrt(self.__B**2/(x_c[i]**6)*(1-0.75*S[i]))
            
            rho = B**2/(4*np.pi*v**2)
            v_A.append(v)
            B_c.append(B)
            rho_c.append(rho)
            
        return S, v_A, B_c, rho_c
    
    
    def plot_Alfven_field(self, S_num, filepath = None):
        
        S, v_A, B_c, rho_c = self.crit_field(S_num)
        
        fig, ax1 = plt.subplots()
        ax1.semilogy(S, rho_c, '.-', color = 'tab:green')
        ax1.set_xlabel(r'Colatitude, $S$')
        ax1.set_ylabel(r'Crit. density, $\rho_c$ (gcm$^{-3}$)')
        ax1.tick_params(axis='y')
        
        ax2 = ax1.twinx()
        ax2.semilogy(S, B_c, '.-', color = 'tab:red')
        ax2.set_ylabel('Crit. mag. field strength, $B_c$ (G)', color = 'tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        ax1.set_title(r'$M =$'+'{}'.format(self.__mass/M_sun) + r'$M_{\odot}$, ' + \
                      r'$R =$'+'{}'.format(self.__radius/R_sun) + r'$R_{\odot}$, ' + \
                      r'$T =${}'.format(self.__period) + r' days, ' + \
                          r'$\bar B = ${}'.format(self.__B) + 'G')
        fig.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi = 200)
                    

    def poly_fits(self, S_num, fits = False):
        """
        Fit 9th order polynomials to the x_c data and find polynomial derivative.
        
        Returns:
        --------
        P : 9TH ORDER NUMPY POLYNOMIAL OBJECT FITTED TO LN(X_C) VS SQRT(S)
        
        
        P_dash : NUMPY POLYNOMIAL OBJECT GIVING DERIVATIVE P'
        
        """
        
        # calculate x_c and U_c
        S, xc, Uc = self.calculate_xc(S_num)
        
        # calculate magnetic field, Alfven velocity and density
        _, v_A, B_c, rho_c = self.crit_field(S_num)

        # converting from high precision to regular precision:
        sqrt_S = [np.sqrt(float(S[i])) for i in range(len(xc))] 
        xc = [float(xc[i]) for i in range(len(xc))] 
        rho_c = [float(rho_c[i]) for i in range(len(rho_c))]
        v_A = [float(v_A[i]) for i in range(len(v_A))]

        # log of critical density multiplied by Alfven velocity
        rho_c_v_A = np.array([rho_c[i]*v_A[i] for i in range(len(rho_c))])

        
        deg = 9 # 9th degree polynomials
        z, res, _, _, _ = np.polyfit(sqrt_S, xc, deg, full = True)
        z2, res2, _,_,_ = np.polyfit(sqrt_S, rho_c_v_A, deg, full = True)
        P = np.poly1d(z) 
        Q = np.poly1d(z2)
        P_dash = np.polyder(P)
        
        self._P = P
        self._Q = Q
        self._P_dash = P_dash
        
        if fits:
            return sqrt_S, xc, rho_c_v_A, P, P_dash, res
        else:
            return P, rho_c_v_A, P_dash
        
    def Jdot_radial(self):
        """
        Reihners and Mohanty (2012) model
        """
        C = 2.66*10**3
        
        omega_crit = 2*np.pi/(8.5*24*60*60)
        
        if self.__omega < omega_crit:
           return C*(self.__omega/omega_crit)**4*self.__omega*(self.__radius**16/self.__mass**2)**(1/3)
       
        else:
            return C*self.__omega*(self.__radius**16/self.__mass**2)**(1/3)
        
    # NEED TO CHANGE THIS FUNCTION
    def plot_poly_fits(self, S_num, filepath = None):
        
        sqrt_S, log_xc, log_rho_c_v_A, P, Q, P_dash, res, res2 = self.poly_fits(S_num, fits = True)
        
        x = np.linspace(min(sqrt_S), max(sqrt_S), 100)
        deg = 9
        
        fig, ax = plt.subplots(2,1, sharex = True)
        ax[0].set_title(r'$log(\kappa) = {}$,'.format(np.round(np.log10(float(self.kappa)), decimals = 2)) + \
            r' $L = {}$,'.format(np.round(float(self.l), decimals = 2)) + \
        r' $\zeta = {}$'.format(np.round(float(self.zeta), decimals = 2)) + '\n Critical radius, '+ r'$x_c$')
        ax[0].plot(sqrt_S, log_xc, '.', color = 'tab:blue')
        ax[0].set_ylabel(r'$\ln(x_c)$')
        # plt.title('Kappa {}'.format(K) +', L {}'.format(np.round(L, decimals = 2)))
        ax[0].plot(x, P(x), color = 'tab:red', label = 'Poly. degree {}'.format(deg) + \
                   ' RMS Error {}'.format(np.round(np.sqrt(res[0]/100), decimals = 5)))
        ax[0].legend()
        
        ax[1].set_title(r'Crit. density $\times$ Alfven velocity, $\rho_cv_{c,A}$')
        ax[1].plot(sqrt_S, log_rho_c_v_A, '.', color = 'tab:blue')
        ax[1].set_xlabel(r'$\sin\theta$')
        ax[1].set_ylabel(r'$\ln(\rho_cv_{c,A})$')
        # plt.title('Kappa {}'.format(K) +', L {}'.format(np.round(L, decimals = 2)))
        ax[1].plot(x, Q(x), color = 'tab:red', label = 'Poly. degree {}'.format(deg) + \
                   ' RMS Error {}'.format(np.round(np.sqrt(res2[0]/100), decimals = 5)))
        plt.legend()
        
        plt.tight_layout()
        
        plt.figure()
        plt.plot(sqrt_S, np.exp(log_rho_c_v_A), 'x-')
        plt.figure()
        plt.plot(sqrt_S, np.exp(log_xc), 'x-')
        if filepath:
            plt.savefig(filepath, dpi = 200)


    def delta_M(self, x):
        
        # x is sin(theta)
        
        return 4*np.pi*self.__radius**2*self._Q(x)*x*(self._P(x)**2 + \
                                0.5*x*self._P_dash(x))/(np.sqrt(1-0.75*x**2))
        
    def delta_J(self, x):
        
         # x is sin(theta)
        
        return 4*np.pi*self.__radius**4*self.__omega*self._P(x)**2*x**3* \
                    self._Q(x)*(self._P(x)**2 + 0.5*x*self._P_dash(x))/ \
                        np.sqrt(1-0.75*x**2)
            
    
    def plot_delta_M(self):
        
        _ = self.poly_fits(100) # to get P, P_dash and rhovca
        
        x = np.linspace(1e-9, np.sqrt(float(self.last_fieldline)), 100)
        
        plt.figure(1)
        plt.plot(x, self.delta_M(x), 'x-')
        plt.ylabel(r'$\Delta \dot{M}$')
        plt.xlabel(r'$\sin\theta$')
        
        plt.figure(2)
        plt.plot(x, self.delta_J(x), 'x-')
        plt.ylabel(r'$\Delta \dot{J}$')        
        plt.xlabel(r'$\sin\theta$')    
    
    def M_dot_J_dot(self, S_num):
        
        # calculate the polynomial fits for this star
        _ = self.poly_fits(S_num)
        
        # calculate mass integral
        M_dot = quad(self.delta_M, 0, np.sqrt(float(self.last_fieldline)))
        J_dot = quad(self.delta_J, 0, np.sqrt(float(self.last_fieldline)))
        
        return M_dot[0], J_dot[0]
            
    def __fieldline_equation(self, x, R, S):
        """
        Equation of a fieldline emergig from S, according to Mestel Eq. (11)
        """
        return np.sqrt((x**4*R**2/S**2)**(1/3) - x**2)
    
    def __radial_fieldline(self, x, m):
        return x*m
 
    def plot_field(self, ax, lim):
        
        S = np.linspace(0.01,1,30)**2
        
    #    fig, ax = plt.subplots(figsize = (6,6))
        
        xc_init, Uc = self.__crit_velocity(S[0])
        # lim = 1.2*self.__fieldline_equation(float(xc_init)*np.sqrt(S[0]), 1, S[0])
        
        ax.set_aspect('equal')
    
        ax.set_ylim(-1,lim)
        ax.set_xlim(-lim,lim)
        x = np.linspace(-lim,lim,10000)
    
        circ= plt.Circle((0,0), 1, color = 'white', zorder = 100, ec = 'k')
        ax.add_artist(circ)
        
        x_alf = []
        y_alf = []
        for i in range(len(S)):
            # if the fieldline is open
            if S[i] < self.last_fieldline:
                print('Last fieldline, S = ', self.last_fieldline)
                # calculate xc
                xc, Uc = self.__crit_velocity(S[i])
                
                # cartesian coordinates of the Alfven point
                last_x = float(xc)*np.sqrt(S[i])
        
                last_y = self.__fieldline_equation(float(xc)*np.sqrt(S[i]), 1, S[i])
        
                x_alf.append(last_x)
                y_alf.append(last_y)        
        
                x_list = np.array([i for i in x if abs(i) < last_x])
                x_list2 = np.array([i for i in x if i > last_x])
        
                # plot the full dipolar fieldlines
                ax.plot(x, self.__fieldline_equation(x, 1, S[i]), color = 'navy', alpha = 0.1)
                ax.plot(x, -self.__fieldline_equation(x, 1, S[i]), color = 'navy', alpha = 0.1)
            
                # plot dipolar fieldlines up to the Alfven point
                ax.plot(x_list, self.__fieldline_equation(x_list, 1, S[i]), color = 'navy', linewidth = 1)
                ax.plot(x_list2, self.__radial_fieldline(x_list2, last_y/last_x), color = 'navy', linewidth = 1)
                ax.plot(-x_list2, self.__radial_fieldline(-x_list2, -last_y/last_x), color = 'navy', linewidth = 1)
                ax.plot(last_x, last_y, 'x', color = 'black', markersize = 3)
                ax.plot(-last_x,last_y, 'x', color = 'black', markersize = 3)
            
            else:
                ax.plot(x, self.__fieldline_equation(x, 1, S[i]), color = 'navy', linewidth = 1)
                ax.plot(x, -self.__fieldline_equation(x, 1, S[i]), color = 'navy', linewidth = 1)
        
        x_alf = np.array(x_alf)
        y_alf = np.array(y_alf)
        
        return ax 


class Full_Evolution(Star):
    """

    """
    
    def __init__(self, initial_mass, initial_period, P_crit, B_crit):
        # OPEN PARAMETERS 
        #------------------------------ 
        self.B_crit = B_crit
        self.P_crit = P_crit
        self.omega_crit = 2*np.pi/(self.P_crit*24*60*60)
        self.a = 1.5
        self.t_lock = 5*10**6

        self.mass0 = initial_mass
        self.mass_rat = np.round(initial_mass/M_sun, decimals = 6)
        self.period0 = initial_period
        self.omega0 = 2*np.pi/(self.period0*24*60*60)
        
        # FIND THE EVOLUTIONARY TRACK & GET FUNCTION FOR dR/dt
        # (this is only done once for the entire evolution)
        self.t_init, self.t_final, self.R_init, self.R_dot_func, self.R_func = self.R_dot()
        
        # initial age IN YRS from evolutionary tracks
        self.t0 = 10**self.t_init               
        # intial radius IN CM from evolutionary tracks
        self.radius0 = self.R_init*R_sun             
        # initial angular mometun IN GCM^2/SEC
        self.J0 = 2/5*self.mass0*self.radius0**2*self.omega0 
        # initial B-field strength
        self.B0 = self.B0evol_B_field(self.omega0) # intial magnetic field 

        
    def R_dot(self, plot = False):
        """
        d(log(R))/d(log(t)) calculation. Return a functional for this derivative 
        from the tracks data.
        If the exact stellar mass given is present in Barraffe tracks, uses 
        those values. Otherwise, looks at neighbouring M values and 
        interpolates linearly to get the evolutionary track.
        
        Returns:
        --------
        min(t) : INITIAL AGE OF THE STAR TO START THE EVOLUTION
        
        max(t) : FINAL AGE OF STAR TO FINISH THE EVOLUTION
            
        r[0] : INTIAL RADIUS OF THE STAR TO START THE EVOLUTION
            
        der : SPLINE OF THE DERIVATIVE, CAN BE USED TO FIND DERIVATIVE AT ANY AGE
        """   
        df = pd.read_csv('BHAC15_tracks_usable.csv', delim_whitespace = True)
        
        M_unique = pd.unique(df['M/Ms'])
        
        # FIND NEIGHBOURING VALUES OF M IN THE TRACKS (M1 and M2)
        M_value = find_nearest(M_unique, self.mass0/M_sun)
        
        t1, t2 = None, None
        r1, r2 = None, None
        M1, M2 = None, None
        
        if M_value[1] == np.round(self.mass_rat, decimals = 6):
            
            t, radius = [], []
            for i in range(len(df['M/Ms'])):
                if df['M/Ms'][i] == M_value[1]:
                    t.append(df['logt(yr)'][i])
                    radius.append(df['R/Rs'][i])
                           
            spline = CubicSpline(t, radius)
            der = spline.derivative()
            
        else:
            if M_value[1] < self.mass_rat:
                M1, M2 = M_value[1], M_unique[M_value[0]+1]
            else:
                M1, M2 = M_unique[M_value[0]-1], M_value[1]
                
            # GET THE DATA FOR M1 and M2
            t1, t2 = [], []
            r1, r2 = [], []
            
            for i in range(len(df['M/Ms'])):
                if df['M/Ms'][i] == M1:
                    t1.append(df['logt(yr)'][i])
                    r1.append(df['R/Rs'][i])
                if df['M/Ms'][i] == M2:
                    t2.append(df['logt(yr)'][i])
                    r2.append(df['R/Rs'][i])
            
            # FIND THE FORM OF THE R VS T DATA FOR M1 and M2
            spline1, spline2 = CubicSpline(t1, r1), CubicSpline(t2,r2)
            
            # LINEAR INTERPOLATION BETWEEN THE TWO SETS OF DATA AT A REGULAR INTERVALS
            # split the t range into 432 (same as all BHAC15 data) regularly spaced t and 
            # interpolate linearly between the values at this t for m1 and m2
            t = np.linspace(min(t2), max(t2), 432)
            radius = []
            for i in range(len(t)):
                lin = interp1d([M1,M2], [float(spline1(t[i])), float(spline2(t[i]))], kind = 'linear')
                radius.append(float(lin(self.mass_rat)))
              
            # SPLINE INTERPOLATION OF THE ACQUIRED DATA POINTS AND FINDING DERIVATIVES
            spline = CubicSpline(t, radius)
            der = spline.derivative()
            
        if plot:
            if t1:
                return min(t), max(t), t, radius, M1, M2, self.mass_rat, t1, r1, t2, r2
            else:
                return min(t), max(t), t, radius, False, False, self.mass_rat, False, False, False, False,

        else:
            return min(t), max(t), radius[0], der, spline    
        
        
    def plot_evol_track(self, filepath = None):
        
        min_t, max_t, t, radius, M1, M2, M_star, t1, r1, t2, r2 = self.R_dot(plot = True)
        x = np.linspace(min_t, max_t, 432)

        plt.figure()
        
        if t1:
            plt.plot(t1, r1, '-', label = r'$M = {}$'.format(M1) + r'$M_{\odot}$')
            plt.plot(t2, r2, '-', label = r'$M = {}$'.format(M2) + r'$M_{\odot}$')
        plt.plot(t, radius, 'x-', label = r'$M = {}$'.format(M_star) + r'$M_{\odot}$ (interpolated)')  
        plt.title(r'Evolutionary track (BHAC15)')  
        plt.xlabel(r'$log(t)$, yr')
        plt.ylabel(r'$R/R_{\odot}$')
        plt.legend()
        plt.tight_layout()
            
        if filepath:   
            plt.savefig(filepath, dpi = 200)  
        
        return t, radius
            
            
    def evol_B_field(self, omega):
    
        if omega >= self.omega_crit:
            return self.B_crit
        
        else:
            return self.B_crit*(omega/self.omega_crit)**self.a   
        
    def radius_at_age(self, age):
        """
        Determine the radius at some age from the Barraffe tracks.
        """
        _, _, _, _, spline  = self.R_dot()
        rad = spline(np.log10(age))*R_sun
        
        return rad
        
    def evolve(self):
        """
        Evolved the star of given mass and period according to Baraffe
        evolutionary tracks (radius is determined from the tracks).

        """
        t_lock = 5*10**6
        
        
        # create a directory for plots
        try:
            os.mkdir('evolution_data/plots/M{}'.format(np.round(self.mass0/M_sun, \
                                                            decimals = 2)))
        except OSError:
            print()
        
        start = time.time()
        
        # set up the data frame 
        M_array = []
        R_array = []
        P_array = []
        t_array = []
        M_dot_array = []
        J_dot_array = []
        B_array = []
        j = []
        sm = []
        L = []
        Kappa = []
        
        # initial parameters to commence evolution
        mass = self.mass0
        radius = self.radius0
        period = self.period0
        omega = self.omega0
        B = self.B0
        J = self.J0
        t = self.t0
        
        S_num = 100
        
        # number of steps
        i = 0
        first = True
        # interate until the final age given in the tracks
        while t < 10**self.t_final:
            print('----------')
            # initialize a Star class with the given parameters
            star = Star(mass, radius, period, B)
            i+=1
            print('age', t/10**6, 'Myr')
            
            # CALCULATE TIME STEP
            # find Helmholtz-Kelvin timescale 
            t_KH = t*radius/R_sun*np.log(10)/abs(self.R_dot_func(np.log10(t)))
            # time step in years
            dt = t_KH/10 
            
            if dt > t:
                dt = t
            
            # CALCULATE MASS AND ANG. MOMENTUM OUTFLOW
            M_dot, J_dot = star.M_dot_J_dot(S_num)
            M_dot_array.append(M_dot)
            J_dot_array.append(J_dot)
            
            # UPDATE PARAMETERS
            # UPDATE TIME (in years)
            t = t + dt
            mass = mass + M_dot*dt*60*60*24*365
            radius = self.R_func(np.log10(t))*R_sun
            
            if t <= t_lock:   
                J = 2/5*mass*radius**2*omega
                # omega and period stay constant, while radius decreases 
                # so J decreases
                

            else:
                if first:
                    # first non-locked iteration:
                    # we take angular 
                    # momentum as the one at t_lock
                    rad_tlock = self.radius_at_age(self.t_lock)
                    J = 2/5*mass*rad_tlock**2*omega
                    first = False

                J = J + J_dot*dt*60*60*24*365
                omega = J/(2/5*mass*radius**2)               # new omega
                period = 2*np.pi/(omega*24*60*60)            # new time period
            B = self.evol_B_field(omega)
                
            M_array.append(mass/M_sun)
            R_array.append(radius/R_sun)
            P_array.append(period)
            t_array.append(t)
            B_array.append(B)
            L.append(float(star.l))
            Kappa.append(float(star.kappa))
            sm.append(float(star.last_fieldline))
            j.append(J)
            print('Jtotal: ', J)
            print('Jdot: ', J_dot*dt*60*60*24*365)
            print('Last line: ', star.last_fieldline)
            print('age: ', t/10**6, 'Myr')
            print('R: ', radius/R_sun)
            print('period: ', period)
            
        # print('----------')  
        print('TIME TAKEN', time.time() - start, 'sec')
        print('TOTAL STEPS', i)
        
        # CSV FILE WITH DATA
        data = {'M/Ms' : M_array,
                'R/Rs' : R_array,
                'Period' : P_array,
                't (yrs)' : t_array,
                'Mdot (g/s)' : M_dot_array,
                'Jdot (gcm^2/s^2)' : J_dot_array,
                'J' : j,
                'B0' : B_array,
                'Last open fieldline, Sm' : sm,
                'L' : L,
                'Kappa' : Kappa
                }
        
        df = pd.DataFrame(data)
        df.to_csv('evolution_data/evolutionM{}_Pin{}_Pcrit{}_Bcrit{}.csv'.format(self.mass_rat,\
                                self.period0, self.P_crit, self.B_crit), index = False)
        self.evolution_data = df
    
    def plot_Mdot_Jdot(self):
        
        # PLOTS OF EVOLUTION

        df = self.evolution_data
        
        fig, ax = plt.subplots(2, 2, figsize = [8,6])
        
        # PERIOD
        ax[0,0].set_title('Period evolution')
        ax[0,0].loglog(df['t (yrs)'], df['Period'], '.-', color = 'tab:red')
        ax[0,0].axvline(x = self.t_lock, color = 'k')
        ax[0,0].set_xlabel('Age, yrs')
        ax[0,0].set_ylabel('$P$, days')
        
        # MASS OUTFLOW
        ax[0,1].set_title('Mass outflow evolution')
        Mdot = np.array([-i for i in df['Mdot (g/s)']])
        Mdot = Mdot/M_sun*365*24*60*60
        ax[0,1].axvline(x = self.t_lock, color = 'k')
        ax[0,1].loglog(df['t (yrs)'], Mdot , '.-', color = 'tab:red')
        ax[0,1].set_ylabel(r'$\dot{M}$, $M_{\odot}$/yr')
        ax[0,1].set_xlabel('Age, yrs')
        
        # ANG MOMENTUM OUTFLOW
        ax[1,1].set_title('Ang. mometum outflow evolution')
        Jdot = np.array([-i for i in df['Jdot (gcm^2/s^2)']])
        ax[1,1].axvline(x = self.t_lock, color = 'k')
        ax[1,1].loglog(df['t (yrs)'], Jdot , '.-', color = 'tab:red')
        ax[1,1].set_ylabel(r'$\dot{J}$, gcm$^2$/s$^2$')
        ax[1,1].set_xlabel('Age, yrs')
        
        # B FIELD
        ax[1,0].set_title('Mag. field strength evolution')
        ax[1,0].axvline(x = self.t_lock, color = 'k')
        ax[1,0].loglog(df['t (yrs)'], df['B0'] , '.-', color = 'tab:red')
        ax[1,0].set_ylabel('$B_0$, Gauss')
        ax[1,0].set_xlabel('Age, yrs')
        plt.tight_layout()
        plt.savefig('evolution_data/plots/M{}'.format(self.mass_rat) + '/evolution_P{}_days.png'.format(self.period0), dpi = 150)

        
    def plot_field_evolution(self):
        
        lim = 20
        
        df = self.evolution_data
        
        fig, ax2d = plt.subplots(3, 3, squeeze=False, figsize = (8,8))
        ax = ax2d.flatten()
        
        step = len(df)//9
        if len(df)%9 == 0:
            rows = np.arange(0,len(df), step)
        else:
            rows = np.arange(3,len(df)+3, step)
        
        for i in range(9):
            S = Star(M = df['M/Ms'][rows[i]]*M_sun, R = df['R/Rs'][rows[i]]*R_sun, \
                     P = df['Period'][rows[i]], B = df['B0'][rows[i]])
            print(S.last_fieldline)
            S.plot_field(ax[i], lim)
            
            ax[i].set_title('t = {}Myr'.format(np.round(df['t (yrs)'][rows[i]]/10**6, decimals = 2)))
            
        fig.suptitle('{}'.format(self.mass0/M_sun) + r'$M_{\odot}$,' + \
                     r'  Init. period = {} day(s)'.format(self.period0))
        fig.tight_layout()
        plt.savefig('evolution_data/plots/M{}'.format(self.mass_rat) + \
                    '/field_evolution_P{}_days.png'.format(self.period0), \
                        dpi = 150)

    def get_period_at(self, age, only_contraction = False):
        """
        Perform a spline interpolation of age vs period data, and hence 
        estimate the value of period at a given age.
        """
        #df = pd.read_csv('evolution_data/evolutionM0.3_Pin10_Pcrit8.5_Bcrit1000.csv')
        
        df = self.evolution_data
        
        # if we ignore the angular momentum outflow, just contraction
        if only_contraction:
            
            rad_tlock = self.radius_at_age(self.t_lock)
            # print('rad_lock',rad_tlock)
            omega_tlock = 2*np.pi/(self.period0*24*60*60)
            # print('omegtlock', omega_tlock)
            
            # ang.mom at locking
            J_tlock = 2/5*self.mass0*rad_tlock**2*omega_tlock
            # print('J lock', J_tlock)
            # print('J0', self.J0)
            
            rad = self.radius_at_age(age)
            # assuming ang.mom. and mass constant, omega at age we want is
            omega = J_tlock/(2/5*self.mass0*rad**2)
            period = 2*np.pi/(omega*24*60*60)   

            return period
        
        else:
            spline = CubicSpline(df['t (yrs)'], df['Period'])
            return spline(age)
















