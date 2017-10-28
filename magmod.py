from __future__ import division
import numpy as np
from numpy import sqrt, sin, cos
import sys
from scipy import integrate
import scipy.special as sp



from homogen import hom, cosmo as cosmologies
cosmo = cosmologies.std
c=hom.c


h = cosmo['h']; sigma8 = cosmo['sigma_8']; n_s = cosmo['ns']; Omega_m = cosmo['omega_M_0']; H_0 = cosmo['H_0']; Omega_lambda = cosmo['omega_lambda_0']
H_z = hom.H_z; E_z = hom.E_z

#Omega_bar=f_bar*Omega_m
Omega_bar=0.047 #baryons
Omega_HI=2.45*10**-4/h #where did I get that from?



nu21emit = 1420.406
def nutoz21( nuobs ):
    return (nu21emit - nuobs) / nuobs

def ztonu21( z ):
    return nu21emit / (1+z)

###################################################################################################################################
#Growth factor from Aseem via Sebastian
###################################################################################################################################
def D_z1(z):
    '''calculates Growth rate (eq 7.77 in Dod) with hypergeom. fct.'''

    a = 1.0/(1+z)
    acube = a**3
    hbyh0 = E_z(z)

    g = hbyh0/np.sqrt(Omega_m)*a**2.5*sp.hyp2f1(5.0/6,1.5,11.0/6,-acube*(1.0/Omega_m-1))

    return g

def D_z(z):
	return D_z1(z)/D_z1(0.)

def T_obs(z):
	return 44*(Omega_HI*h/(2.45*10**-4))*(1+z)**2/E_z(z) #myK

def W_tophat(z,zmin,zmax):
    """arguments: z, zmin, zmax
    z can be scalar or array"""
    if type(z)==np.ndarray:
    	res=np.zeros(z.shape[0])
    	res[np.where((z>=zmin)&(z<=zmax))]=1/(zmax-zmin)
    	return res
    if (z>=zmin)&(z<=zmax):
    	return 1/(zmax-zmin)
    return 0

###################################################################################################################################
#BBKS stuff:
###################################################################################################################################
constBBKS=1 #to be changed later

def W_k(k):
	return 3/k**3*(np.sin(k)-k*np.cos(k))

def q(k): ##LOOK
	Gam=Omega_m*h**2*np.exp(-Omega_bar*(1+(2*h)**0.5/Omega_m))
	return k/Gam # this way consistent with Battye...[1/Mpc]

def T_BBKS(k):
	q_ev=q(k)
	return np.log(1+2.34*q_ev)/(2.34*q_ev)*(1+3.89*q_ev+(16.1*q_ev)**2+(5.46*q_ev)**3+(6.71*q_ev)**4)**(-1./4)

def P_cdm_BBKS(k): #No z dependence
	return constBBKS*k**n_s*T_BBKS(k)**2

def T_HI_BBKS(k,z):
	return (T_obs(z)**2*k**3*P_cdm_BBKS(k)/(2*np.pi**2))**0.5

def sigma_BBKS(R):
	return (1./(2*np.pi**2)*integrate.quad(lambda k: P_cdm_BBKS(k)*W_k(k*R)**2*k**2,0.,np.inf,full_output=1)[0])**0.5
constBBKS=(sigma8/sigma_BBKS(8/h))**2
print 'sigma_8_BBKS =', sigma_BBKS(8/h)


###################################################################################################################################
#comoving distance from aseem via sebastian
###################################################################################################################################
def rCom(z):
    '''comoving distance without numerical integration
	Flat LCDM comoving distance to redshift z, output in Mpc.
        rCom(z) = int dz / H(z)
	'''
    OmbyOl = Omega_m/(Omega_lambda)
    out = ((1+z)*sp.hyp2f1(1/3.0,0.5,4/3.0,-(1+z)**3*OmbyOl)
           - sp.hyp2f1(1/3.0,0.5,4/3.0,-OmbyOl))
    out *= 2998/np.sqrt(Omega_lambda)
    return out/h
