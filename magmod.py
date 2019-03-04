from __future__ import division
import os
import numpy as np
import healpy as hp
from numpy import pi, sin, cos, tan, e, arctan, arcsin, arccos, sqrt
import sys
from scipy import integrate
import scipy.special as sp
from scipy.interpolate import interp2d, interp1d
from time import clock #timing
from astropy.io import ascii as ii


from lf_photometric import s_magbias, bias as galaxy_bias, nz_distribution


from homogen import hom, cosmo as cosmologies
from magbias_experiments import SKA_zhangpen, CLAR_zhangpen, SKA, hirax, LSST, n
cosmo = cosmologies.std
c=hom.c

#unit conversions
Mpctom=3.0857*10**22
mtoMpc=1/Mpctom
INF_NOISE = 1e200 # Very large finite no. used to denote infinite noise


HRS_MHZ = 1e6*60**2 #conversion, 1 h = 1e6*60**2 MHz^-1

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
	return 1e-3*44*(Omega_HI*h/(2.45*10**-4))*(1+z)**2/E_z(z) #mK

def Tb_redbook(zc):
    return 0.0559+0.2324*zc-0.024*pow(zc,2)


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



############################################################################
############################################################################
#functions for the cross correlation power spectra
#
############################################################################
############################################################################


#preliminary, to be changed!!
# bHI = 1
# bgal = 1

#camb power spectrum:
if os.path.exists('/mnt/lustre/users/awitzemann/'): #this means we are chpc
    pknl_root = "/mnt/lustre/users/awitzemann/up/MagBias/PkNL/"
else:
    pknl_root = "/Users/amadeus/Documents/PhD/work/MagBias/PkNL/"
filelist =  np.sort(os.listdir(pknl_root))
# print "CHECK IF THIS IS IN ORDER:"
# print filelist
zc_pk = np.linspace(0,4,len(filelist))
# print zc_pk

k_pk = np.loadtxt(pknl_root + filelist[0], unpack = True)[0]

pknl_arr = np.array([ np.loadtxt(pknl_root + fi, unpack = True)[1] for fi in filelist ])

pknl_int = interp2d( k_pk, zc_pk, pknl_arr)

def pknl(kk, zz):
    res = pknl_int(kk/h, zz)/h**3
    if len(res) == 1:
        return res[0]
    else:
        return res


# difficult to do quick integration because integration limit appears inside the integral. Therefore we split it up!
#lensing kernel from Alkistis' notes:
def g_old(z, zb, dzb, NINT = 10000, S_G = "old fit", MAXMAG = False): #does not include sg inside the integral
    """compare to my (handwritten) notebook 2 entry for 9/3/17
    zb is background redshift"""
    if S_G != "old fit":
        raise ValueError
    if MAXMAG != False:
        raise ValueError
    print "using old lensing kernel"
    zmin = zb - dzb
    zmax = zb + dzb
    z = np.atleast_1d(z)
    zintmin = 1e-3 #lower integration limit, z must not be smaller than this
    if (z>zmax).any() or (z<zintmin).any():
        raise ValueError("z either lies behind the background or is too small! z = {} to {}".format(np.amin(z), np.amax(z)))

    zint = np.linspace(zmax, zintmin, NINT) #going reverse, multiply integral with -1
    Warray = W_tophat(zint, zmin, zmax)
    rarray = rCom(zint)
    Wbyrarray = Warray/rarray

    integrand1 = Warray
    integrand2 = Wbyrarray

    integral1 = -integrate.cumtrapz( integrand1, zint, initial = 0)
    integral2 = -integrate.cumtrapz( integrand2, zint, initial = 0)

    val1 = interp1d(zint[::-1], integral1[::-1], kind='linear', bounds_error=False)
    val2 = interp1d(zint[::-1], integral2[::-1], kind='linear', bounds_error=False)

    rComtab = rCom(z)
    result = rComtab*(val1(z) - rComtab*val2(z))
    return result


def g_tophat(z, zb, dzb, NINT = 10000, S_G = "LSST", MAXMAG = False): #new standard to use cumtrapz, also with sg inside the integration
# def g(z, zb, dzb, NINT = 500, S_G = "LSST"): ############TEST:QUICKER INTEGRATION#############new standard to use cumtrapz, also with sg inside the integration
    """compare to my (handwritten) notebook 2 entry for 9/3/18
    zb is background redshift"""

    #temporarily forcing the use of MAXMAG:
    if type(MAXMAG) == bool and MAXMAG == False:
        raise ValueError("please give a value to MAXMAG!")


    elif S_G == "LSST":
        sgfunc = sg
    elif S_G == "old fit":
        sgfunc = sg_old
    else: raise ValueError

    zmin = zb - dzb
    zmax = zb + dzb
    z = np.atleast_1d(z)
    zintmin = 1e-3 #lower integration limit, z must not be smaller than this
    if (z>zmax).any() or (z<zintmin).any():
        raise ValueError("z either lies behind the background or is too small! z = {} to {}".format(np.amin(z), np.amax(z)))

    zint = np.linspace(zmax, zintmin, NINT) #going reverse, multiply integral with -1
    # zint = np.linspace(zmax, zmin, NINT) #going reverse, multiply integral with -1
    # Warray = W_tophat(zint, zmin, zmax)
    if type(MAXMAG) == bool and MAXMAG == False:
        Warray = W_tophat(zint, zmin, zmax) * (5 * sgfunc(zint, MAXMAG = MAXMAG) -2 )
    else:
        Warray = W_dndz(zint, zmin, zmax, MAXMAG) * (5 * sgfunc(zint, MAXMAG = MAXMAG) -2 )

    # Warray = W_tophat(zint, zmin, zmax) * (5 * sgfunc(zb) -2 ) #assuming that sgfunc is constant with redshift. THIS IS A TEST
    rarray = rCom(zint)
    Wbyrarray = Warray/rarray

    integrand1 = Warray
    integrand2 = Wbyrarray

    integral1 = -integrate.cumtrapz( integrand1, zint, initial = 0)
    integral2 = -integrate.cumtrapz( integrand2, zint, initial = 0)

    val1 = interp1d(zint[::-1], integral1[::-1], kind='linear', bounds_error=False)
    val2 = interp1d(zint[::-1], integral2[::-1], kind='linear', bounds_error=False)
    # val1 = interp1d(zint[::-1], integral1[::-1], kind='linear', bounds_error=False, fill_value = 0)
    # val2 = interp1d(zint[::-1], integral2[::-1], kind='linear', bounds_error=False, fill_value = 0)

    rComtab = rCom(z)
    result = rComtab*(val1(z) - rComtab*val2(z))
    return result

def sg5minus2(z, mstar):
    # return (5 * sg_interp(z, MAXMAG = mstar) -2 )
    return (5 * sg(z, MAXMAG = mstar) -2 )


def g_old(z, zb, dzb, mstar, NINT = 10000): #first we try trapz, later something better
    """compare with handwritten entry in notebook 2 from 14/12/2018"""

    if type(mstar) == bool:
        raise ValueError("Need mstar to be given!")
    zmin = zb - dzb
    zmax = zb + dzb
    z = np.atleast_1d(z)
    if (z>zmin).any():
        raise ValueError("foreground z must not lie within the background")

    zint = np.linspace(zmin, zmax, NINT) #going reverse, multiply integral with -1

# trying the bottom approach, splitting up the integral
    fac1 = sg5minus2(zint, mstar)
    fac2 = W_dndz(zint, zmin, zmax, mstar)


    fac12 = fac1*fac2

    fac3 = 1/rCom(zint)

    integral1 = np.trapz(fac12, zint)
    integral2 = np.trapz(fac12*fac3,zint)

    rcomtab = rCom(z)

    return rcomtab * integral1 - rcomtab**2 * integral2

def g(z, zb, dzb, mstar, NINT = 10000): #first we try trapz, later something better
    """compare with handwritten entry in notebook 2 from 14/12/2018"""

    if type(mstar) == bool:
        raise ValueError("Need mstar to be given!")
    zmin = zb - dzb
    zmax = zb + dzb
    z = np.atleast_1d(z)
    if (z>zmin).any():
        raise ValueError("foreground z must not lie within the background")

    zint = np.linspace(zmin, zmax, NINT) #going reverse, multiply integral with -1

# trying the bottom approach, splitting up the integral
    # fac1 = sg5minus2(zint, mstar)
    # fac2 = W_dndz(zint, zmin, zmax, mstar)


    # fac12 = fac1*fac2
    fac1 = sgng_interp(zint, mstar)
    fac2 = 1 / dndz_norm(zmin, zmax, mstar)
    fac12 = fac1*fac2
    fac3 = 1/rCom(zint)

    integral1 = np.trapz(fac12, zint)
    integral2 = np.trapz(fac12*fac3,zint)

    rcomtab = rCom(z)

    return rcomtab * integral1 - rcomtab**2 * integral2



def g_original(z, zb, dzb, NINT = 10000): #new standard to use cumtrapz, also with sg inside the integration
# def g(z, zb, dzb, NINT = 500, S_G = "LSST"): ############TEST:QUICKER INTEGRATION#############new standard to use cumtrapz, also with sg inside the integration
    """compare to my (handwritten) notebook 2 entry for 9/3/18
    zb is background redshift"""

    zmin = zb - dzb
    zmax = zb + dzb
    z = np.atleast_1d(z)
    zintmin = 1e-3 #lower integration limit, z must not be smaller than this
    if (z>zmax).any() or (z<zintmin).any():
        raise ValueError("z either lies behind the background or is too small! z = {} to {}".format(np.amin(z), np.amax(z)))

    zint = np.linspace(zmax, zintmin, NINT) #going reverse, multiply integral with -1
    # Warray = W_tophat(zint, zmin, zmax)
    Warray = W_tophat(zint, zmin, zmax)
    rarray = rCom(zint)
    Wbyrarray = Warray/rarray

    integrand1 = Warray
    integrand2 = Wbyrarray

    integral1 = -integrate.cumtrapz( integrand1, zint, initial = 0)
    integral2 = -integrate.cumtrapz( integrand2, zint, initial = 0)

    val1 = interp1d(zint[::-1], integral1[::-1], kind='linear', bounds_error=False)
    val2 = interp1d(zint[::-1], integral2[::-1], kind='linear', bounds_error=False)

    rComtab = rCom(z)
    result = rComtab*(val1(z) - rComtab*val2(z))
    return result


def Cl_HIxmag_CAMB(ltable, zf, delta_zf, zb, delta_zb, Nint = 500, MAXMAG = False):
    """ltable, zf foreground redshift, zb background redshift,
    delta_zf foreground redshift widht, Nint integration steps"""


    #   NOW NEW: INCLUDING T_OBS AS WE SHOULD
    fac = 3/2 *(H_0/c)**2 * Omega_m #no square on (H_0/c) because it cancels out
    zmin = zf - delta_zf
    zmax = zf + delta_zf
    ztab = np.linspace(zmin, zmax, Nint) #do checks on this! should be 0 to inf

    integrand=np.zeros([len(ztab),len(ltable)])
    for il in range(len(ltable)):
        ell = ltable[il]
        pknltab = np.array([pknl(( ell)/rCom(zzz), zzz) for zzz in ztab])
        # integrand[:,il] = (1+ztab) * bHI(ztab) * T_obs(ztab) * W_tophat(ztab, zmin, zmax) * g(ztab, zb, delta_zb, S_G = S_G) \
        # / rCom(ztab)**2 * pknltab
        integrand[:,il] = (1+ztab) * bHI(ztab) * Tb_redbook(ztab) * W_tophat(ztab, zmin, zmax) * g(ztab, zb, delta_zb, MAXMAG) \
        / rCom(ztab)**2 * pknltab

        #old and slow ways to calculate the same thing:
#             integrand[:,il]= np.array([(1+zzz) * W_tophat(zzz, zfmin, zfmax) * g(zzz, zb, dzb) \
#                                        / rCom(zzz)**2 * pknl(( ell)/rCom(zzz), zzz) for zzz in ztab])
#         integrand[:,il]= np.array([(1+zzz) * W_tophat(zzz, zfmin, zfmax) * g(zzz, zb, dzb) * rCom(zzz) * pknl(( ell)/rCom(zzz), zzz) for zzz in ztab]) #different factor of chi than in alkistis' notes (units are wrong there)
    result= fac * np.trapz(integrand,ztab,axis=0)
    return result

def Cl_HIxmag_CAMB_old(ltable, zf, delta_zf, zb, delta_zb, Nint = 500):
    """ltable, zf foreground redshift, zb background redshift,
    delta_zf foreground redshift widht, Nint integration steps"""


    #note there is no T_obs factor because we want to compare with galaxy case
    fac1 = 3/2 *(H_0/c)**2 * Omega_m #no square on (H_0/c) because it cancels out
    fac2 = 1 * (5*sg_old(zb) - 2) #we take sg out of the integral

    zmin = zf - delta_zf
    zmax = zf + delta_zf
    ztab = np.linspace(zmin, zmax, Nint) #do checks on this! should be 0 to inf

    integrand=np.zeros([len(ztab),len(ltable)])
    for il in range(len(ltable)):
        ell = ltable[il]
        pknltab = np.array([pknl(( ell)/rCom(zzz), zzz) for zzz in ztab])
        integrand[:,il] = (1+ztab) * W_tophat(ztab, zmin, zmax) * g_original(ztab, zb, delta_zb) \
        / rCom(ztab)**2 * pknltab


    result= fac1 * fac2 * np.trapz(integrand,ztab,axis=0)
    return result



def C_l_HIHI_CAMB(ltable,zmin,zmax, Nint = 500):
    """arguments: ell array, zmin, zmax
    returns: Cl as array"""
    ztable = np.linspace(zmin, zmax, Nint)
    integrand=np.zeros([len(ztable),len(ltable)])
    for l in range(len(ltable)):
        #   NOW NEW: INCLUDING T_OBS AS WE SHOULD
        # integrand[:,l]= np.array([E_z(zzz)*(bHI(zzz) * T_obs(zzz) * W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
        integrand[:,l]= np.array([E_z(zzz)*(bHI(zzz) * Tb_redbook(zzz) * W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
    result=H_0/c*np.trapz(integrand,ztable,axis=0)
    return result

# def C_l_gg_CAMB(ltable,zmin,zmax, experiment = LSST, Nint = 500):
# def C_l_gg_CAMB(ltable,zmin,zmax, Nint = 500):
#     """arguments: ell array, zmin, zmax
#     returns: Cl as array"""
#     ztable = np.linspace(zmin, zmax, Nint)
#     integrand=np.zeros([len(ztable),len(ltable)])
#     for l in range(len(ltable)):
#         integrand[:,l]=np.array([E_z(zzz)*(bgal(zzz) * W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
#     result=H_0/c*np.trapz(integrand,ztable,axis=0)
#     return result

def C_l_gg_CAMB(ltable,zmin,zmax, mstar, Nint = 500):
    """arguments: ell array, zmin, zmax
    returns: Cl as array"""
    ztable = np.linspace(zmin, zmax, Nint)
    integrand=np.zeros([len(ztable),len(ltable)])
    for l in range(len(ltable)):
        integrand[:,l]=W_dndz(ztable, zmin, zmax, mstar)**2 * np.array([E_z(zzz)*(bgal(zzz)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
    result=H_0/c*np.trapz(integrand,ztable,axis=0)
    return result


def DELTA_Cl_HIxmag(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = "CV", MAXMAG = False, nside = 256):
    """zf is mean redshift of foregrounds, dzf is half the width of the bin. Same goes for zb and dzb.
    power_spectra_list needs to be a list [ClHIHI, Clgg, ClHIXmag].
    Example [C_l_HIHI_CAMB, C_l_gg_CAMB, Cl_HIxmag_CAMB].
    SURVEY == CV means cosmic variance limited survey."""


    zfmin = zf - dzf; zfmax = zf+dzf
    zbmin = zb-dzb; zbmax = zb+dzb
    # ClHIHIfunc, Clggfunc, Cl_HIxmagfunc = power_spectra_list
    HIHI, gg, XX = power_spectra_list
    X2 = XX**2


    # d_ell = np.abs(np.mean ( ltable[:-1] - ltable[1:]))
    d_ell = 1
    print "we assume d_ell = 1"
    #perfect survey:
    noisestart = clock()
    if type(SURVEY)==str and SURVEY == "CV":
        Cshot = np.zeros(len(ltable)); N_ell = np.zeros(len(ltable));
        fsky = 1;
    else:
        if type(SURVEY) == list:
            hisurv = SURVEY[0]
            Cshot = shotnoise(zb, dzb, SURVEY[1], MAXMAG = MAXMAG);
        elif type(SURVEY)==dict:
            hisurv = SURVEY
            print "We assume a perfect galaxy survey!"
            Cshot = np.zeros(len(ltable));
        else: raise ValueError("wrong SURVEY")
        fsky = hisurv["S_area"] / (4*np.pi);
        if hisurv["mode"] == "interferometer":
            print "calculating interferometer noise..."
            N_ell = Cl_interferom_noise(ltable, zfmin, zfmax, hisurv)
        elif hisurv["mode"] == "single_dish":
            print "calculating single dish autocorrelation noise"
            N_ell = noise_cls_single_dish(ltable, ztonu21(zf), hisurv, nside) * np.ones( len(ltable) )
            # ell_noise = np.copy(ltable)
        else:
            raise ValueError("dict must contain key 'mode'")
    start = clock()
    # print "noise took {} s".format(start - noisestart)
    # X2 = Cl_HIxmagfunc(ltable, zf, dzf, zb, dzb)**2
    # HIHI = ClHIHIfunc(ltable, zfmin, zfmax)
    # gg = Clggfunc(ltable, zbmin, zbmax)
    # print "it took {} seconds to compute all signals".format(clock() - start)
    num = 2 * (X2 + (HIHI + N_ell) * (gg + Cshot))
    # num = (X2 + (HIHI + N_ell) * (gg + Cshot)) #this is wrong
    denom = (2*ltable+1) * d_ell * fsky
    result = np.sqrt(num/denom)
    return result

def S2N(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = "CV", MAXMAG = False):
    #temporarily forcing the use of MAXMAG if survey not CV:
    if type(MAXMAG) == bool and MAXMAG == False:
        raise ValueError("please use MAXMAG!")
    start = clock()
    delt = DELTA_Cl_HIxmag(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = SURVEY, MAXMAG = MAXMAG)
    mid = clock();
    # signal = power_spectra_list[2](ltable, zf, dzf, zb, dzb)
    signal = power_spectra_list[2]
    end = clock()
    # print "delta took {}s and signal took {} s".format(mid-start, end-mid)
    return signal/delt









###########################################################################
###########################################################################
############### now some functions from the zhang and pen paper
###########################################################################
###########################################################################

#alpha:
def alpha(z, experiment):
    if experiment["Name"] == "SKA" and experiment["S_area"] == 25600:
        #for 25600 sq deg SKA, 4 sigma detection threshold
        alpha_interp_table= [[0, 0.5, 1., 1.5, 2., 2.5, 2.8],[-.5, -0.1, 0.8, 2.2, 4.2, 6.8, 8.8 ]] #[z,alpha-1]
    if experiment["Name"] == "CLAR_zhangpen" and experiment["S_area"] == 160:
        #for 160 sq deg CLAR, 3 sigma detection threshold
        alpha_interp_table= [[0, 0.5, 1., 1.5, 2., 2.5, 2.8], [-.5, -0.25, 0.5, 1.5, 3.1, 5., 6.1]]
    else:
        raise ValueError("wrong experiment given to alpha, not implemented. use sg instead")
    z_alpha_table, alpha_minus_one = alpha_interp_table
    alphatable = np.array(alpha_minus_one) + 1

    alpha_interp = interp1d(z_alpha_table, alphatable, kind = "cubic", bounds_error=None, fill_value=0)
    return alpha_interp(z)

def bgal(z, experiment = LSST):
    return galaxy_bias(z, experiment["rmax"], "all")
    # return 1
def bHI(z): #fit from Alkistis, apparently from the 'red book', but I have not checked yet
    return  0.67 + 0.18*z + 0.05*z**2
    # return  1


# def alpha(z, experiment = LSST, S_G = "LSST"):
#     if S_G == "old fit":
#         sgfunc = sg_old
#     elif S_G =="LSST":
#         sgfunc = sg
#     else: raise ValueError
#     return 5/2 * sgfunc(z, experiment)

# ztab_LSST, sgtab_LSST = np.loadtxt(LSST["sg_file"], unpack = True)
# sginterp_LSST = interp1d(ztab_LSST, sgtab_LSST, kind='cubic', bounds_error=False)
# def sg_fit(z, experiment):
#     if experiment == LSST:
#         # print "quick!"
#         return sginterp_LSST(z) #this is quicker!
#     ztab, sgtab = np.loadtxt(experiment["sg_file"], unpack = True)
#     sginterp = interp1d(ztab, sgtab, kind='cubic', bounds_error=False)
#     return sginterp(z)




ztab_LSST = np.loadtxt(LSST["sgng_file_z"])
magmaxtab_LSST = np.loadtxt(LSST["sgng_file_magmax"] )
sgngtab_LSST = np.loadtxt(LSST["sgng_file"] ) #(5sg-2)ng


sgng_interpolation_func = interp2d(ztab_LSST, magmaxtab_LSST, sgngtab_LSST.T, kind = 'cubic', bounds_error = False)


def sg(z, experiment = LSST, MAXMAG = False):#, force_calc = False):
    """uses the function from David Alonso. 'all' means all galaxy colors, rmax is the magnitude limit of the experiment which needs to be given"""
    # if "sg_file" in experiment.keys() and os.path.exists(experiment["sg_file"]) and not force_calc and type(MAXMAG) == bool and not MAXMAG:
    #     return sg_fit(z, experiment) #quicker! maybe extend this to different rmax cuts
    if type(MAXMAG) == bool and MAXMAG == False:
        rmax_cut = experiment["rmax"]
        print "Using default rmax = {} for experiment".format(rmax_cut)
    else:
        rmax_cut = MAXMAG
        print "Using non default rmax = {} for experiment".format(rmax_cut)
    if type(z) == float:
        return s_magbias(z, rmax_cut, "all")
    z = np.atleast_1d(z)
    # res = [s_magbias(zzz, experiment["rmax"], "all") for zzz in z]
    res = s_magbias(z, rmax_cut, "all")

    #remove the nans:
    res[np.where(np.isnan(res))[0]] = 0
    return np.array(res)

def sgng_interp(z, MAXMAG = False, experiment = LSST):#this is an interpolated version of the (5sg-2)ng, to speed things up!
    z = np.atleast_1d(z)
    if type(MAXMAG) == bool and MAXMAG == False:
        mmax = LSST['rmax']
    else:
        mmax = MAXMAG
    res = sgng_interpolation_func(z,mmax).T
    # if type(z) == np.ndarray and z[-1]<z[0]: #in this case z is an array running from big to small
    #     return res[::-1] #I do not understand exactly why I the interp function needs to be turned around in this case, but it seem like it!
    return res



def sg_old(z, experiment = CLAR_zhangpen, USE_ALPHA = False, MAXMAG = False): #number count slope. Fit taken from eq 23 in 1611.01322v2
    """if USE_ALPHA, then sg is just caluclated from alpha"""
    if not (type(MAXMAG) == bool and MAXMAG == False):
        raise ValueError
    if USE_ALPHA:
        return 2/5*alpha(z, experiment)
    else:
        print "Caution, s_g is implemented without dependence on the experiment"
        n0 = 0.132
        n1 = 0.259
        n2 = -0.281
        n3 = 0.691
        n4 = -0.409
        n5 = 0.152

        res = n0 + n1*z + n2*z**2 + n3*z**3 + n4*z**4 + n5*z**5
        return res





#The total 21 cm flux of HI rich galaxies:
def S21(z, MHI):
    """from eq 1 in Zhang&Pen. returns in mJy, MHI has to be [10^10 solar masses]
    we assume w = 100km/s!"""
    return 0.023 * MHI/10**10 * (c / H_0 / rCom(z))**2 * 1 / (1+z) #rCom should be "comoving angular distance". not sure if that's right

#the system temperature per beam:
def Ssys(z, experiment):
    """experiment is a dict"""
    Tsys = experiment["Tsys"] #[K]
    Aeff = experiment["Aeff"] #[m^2]
    t_int = experiment["t_int"] #[hours] for each FOV!

    fac1 = 0.032 * Tsys / 30
    fac2 = 5e4/Aeff
    fac3 = np.sqrt( (1+z)/t_int )
    return fac1 * fac2 * fac3

n_0 = 0.014 * h**3 #[Mpc^-3]
Mstar = 10**9.55 * h**(-2) #[solar masses]
gamma = 1.2

#schechter function, equation 4:
def n_schechter(M):
    """we set z=0!"""
    fac1 = (M/Mstar)**(-gamma)
    fac2 = np.exp( -M / Mstar)
    return n_0 * fac1 * fac2

def M_HI_min(z, F): #minimum hydrogen mass in galaxy to be detected
    """z redshift, F is flux limit for detection, F = Ssys*n. Derivation of this is based on
    equations 3 and 2, done in amadeus' handwritten notes on 23rd of march
    """
    fac1 = 1.39 * 10**10 * (rCom(z) * H_0 / c)**2 * (1+z)
    fac2 = F / 0.032

    return fac1 * fac2


def M_HI_min_z(z, n, experiment): #minimum hydrogen mass in galaxy to be detected
    """z redshift, experiment is dict. equation 3
    """
    z = np.atleast_1d(z)
    tsy = experiment["Tsys"]
    aef = experiment["Aeff"]
    t = experiment["t_int"]
    fac1 = n * 1.39 * 10**10 #solar masses
    fac2 = (rCom(z)*H_0/c)**2 * (1+z)**(3/2)
    fac3 = np.sqrt( 1/t) * tsy/30 * 5e4/aef
    return fac1*fac2*fac3


def nrho_HI_z(z, n, experiment, Nint = 2000):
    """density of HI galaxies with flux bigger than n * Ssys(z).
    """
    M_HI_lim = M_HI_min_z(z, n, experiment)

    Mint_by_Mstar = np.logspace(2,-6,Nint) #runs backwards
    Mint = Mint_by_Mstar * Mstar
    integrand = n_schechter(Mint)

    integral = integrate.cumtrapz( -integrand, Mint, initial = 0)
    val = interp1d(Mint[::-1], integral[::-1], kind='linear', bounds_error=False)


    rho_lim = val(M_HI_lim)

#     return rho_lim
    return rho_lim/Mstar    #this is not what is written in the paper,
                            #typo! (dM should be dM/M*)


def nrho_HI(z, F, Nint = 2000):
    """density of HI galaxies with flux bigger than F. NOTE:
    """
    F = np.atleast_1d(F)
    M_HI_lim = M_HI_min(z, F)

    Mint_by_Mstar = np.logspace(2,-6,Nint) #runs backwards
    Mint = Mint_by_Mstar * Mstar
    integrand = n_schechter(Mint)

    integral = integrate.cumtrapz( -integrand, Mint, initial = 0)
    val = interp1d(Mint[::-1], integral[::-1], kind='linear', bounds_error=False)


    rho_lim = val(M_HI_lim)

#     return rho_lim
    return rho_lim/Mstar    #this is not what is written in the paper,
                            #but it reproduces their results perfectly. I suspect a typo.. (dM should be dM/M*)




def dNdz(z, n, experiment, Nderiv = 1000, RETURN_PARTS = False):
    """derivative of abundance of 21cm emitting galaxies wrt z.
    for explanation of Fdelta look up N_HI"""
    z = np.atleast_1d(z)
    F = n * Ssys(z, experiment)

    solidangle = experiment["S_area"] * (pi/180)**2

    radius = rCom #comoving
#     radius = lambda zzz: rCom(zzz)/(1+zzz) #physical

    rtab2 = radius(z)**2

    #for derivative:
    zmin = 0.
    zmax = np.amax(z) + 0.2
    ztab_fine = np.linspace( zmin, zmax, Nderiv)
    rtab_fine = radius(ztab_fine)
    drdztab_fine = np.gradient(rtab_fine, ztab_fine)
    drdz = interp1d(ztab_fine, drdztab_fine, kind='linear', bounds_error=False)
    drdztab = drdz(z)

#     ndens = nrho_HI(z, F)
    ndens = nrho_HI_z(z, n, experiment)

    if RETURN_PARTS:
        return solidangle, drdztab, rtab2, ndens

    return solidangle * drdztab * rtab2 * ndens


def N_HI(z, n, experiment, NINT = 2000, Nderiv = 1000):
    """
    This should reproduce fig 1 and 2. z redshift, F flux, both arrays of same length and corresponding values.
    NINT is for numerical integration and Nderiv for derivation in dNdz.
    curiosity Fdelta is there to help derivation wrt F which is not an argument. If Fdelta is number or
    array of length of z it is added to one and multiplied to F = n*Ssys, which is then used for the rest of
    the calculations here.
    """
    z = np.atleast_1d(z)
    zint = np.linspace(0.01, np.amax(z), NINT)

    integrand = dNdz(zint, n, experiment, Nderiv = Nderiv)

    integral = integrate.cumtrapz( integrand, zint, initial = 0)
    val = interp1d(zint, integral, kind='linear', bounds_error=False)
    result = val(z)

    return result



# def G(z, zmin, zmax, nsig, experiment, NINT = 1000, S_G = "LSST"): #kernel of the bg magnification
#     """zmin and zmax are edges of galaxy distribution we're looking at
#     experiment needs to be SKA etc because of alpha"""
#     z = np.atleast_1d(z)
#     zdist = np.linspace(zmin, zmax, NINT)
#     chi = rCom( z )
#     chidist = rCom(zdist) #of galaxy distribution, etiher bg or fg
# #     wmatrix = w_geometry(chi, chidist)
#     ntab = dNdz(zdist, nsig, experiment)
#     Ngal = np.trapz(ntab, zdist)
#     alphatab = alpha(zdist, experiment, S_G)
# #     alphatab = 5/2*sg(zdist) #same thing
#     integrand1 = ntab * 2 * (alphatab -1 )
#     integrand2 = ntab * 2 * (alphatab -1 ) / chidist
#     integral1 = np.trapz( integrand1, zdist)
#     integral2 = np.trapz( integrand2, zdist)
#     return ((1+z)/Ngal) * (chi * integral1 - chi**2 * integral2)

# def Cl_gxmag_CAMB(ltable, zf, delta_zf, zb, delta_zb, nsig, experiment, Nint = 500, S_G = "LSST"):
#     zftab = np.linspace(zf-delta_zf, zf+delta_zf, Nint)
#     zbmin = zb - delta_zb
#     zbmax = zb + delta_zb
#     rtab = rCom(zftab)
#     ntab = dNdz(zftab, nsig, experiment)
#     Ngal = np.trapz(ntab, zftab)
#     integrand=np.zeros((len(zftab),len(ltable)))
#     for il in range(len(ltable)):
#         ell = ltable[il]
#         pknltab = np.array([pknl(( ell)/rCom(zzz), zzz) for zzz in zftab])
#         #we divide by r**3 because we go from Delta_m to P_m
#         integrand[:,il]= 1/rtab**3 * pknltab * G(zftab, zbmin, zbmax, nsig, experiment, S_G = S_G) * dNdz(zftab, nsig, experiment) * rtab
#     fac1 = 3 * Omega_m * H_0**2  / c**2
#     fac2 = np.pi**2#/ ltable**3 #we use k^3 Delta_m = l^3/r^3 Delta_m = P_m
#     fac3 = 1 / Ngal
#     result= fac1 * fac2 * fac3 * np.trapz(integrand,zftab,axis=0)
#     return result * h**6



############################################################
############################################################
############calculate beam and noise:
############################################################
############################################################
def pix_noise(exp_dict, dnu, nside = 256):
    """calculates the pixel noise from a dict with experiment parameters and the frequency."""
    #load dict:
    NPIX = hp.nside2npix(nside)
    t_system = exp_dict["Tsys"]
    nbeams = exp_dict["Nbeam"]
    time_tot = exp_dict["t_int"]
    ndish = exp_dict["Ndish"]
    fsky = exp_dict["S_area"] / (4 * np.pi)
    #calculate the noise RMS :
    DTpix = t_system / np.sqrt(nbeams * time_tot * HRS_MHZ * dnu * ndish / (NPIX * fsky))  #noise RMS per pixel
    return DTpix

def beam_FWHM(exp_dict, mean_nu):
    """returns the beam fwhm of that experiment in the frequency bin"""
    #calculate the beam FWHM:
    mean_lam = c / 1e6 / mean_nu #[m]
    Ddish = exp_dict["Ddish"]

    theta_fwhm = 1.22 * mean_lam / Ddish
    return theta_fwhm

def noise_cls_single_dish(ltab, nu, exp_dict, nside): #taken from amadeus' master thesis eq. 18
    sigpix = pix_noise(exp_dict, nu, nside = nside)
    sigb = beam_FWHM(exp_dict, nu) / np.sqrt( 8 * np.log(2) )
    oneover_W_ell = np.exp( ltab**2 * sigb**2) #beam smoothing function.
    ompix = 4 * np.pi / hp.nside2npix(nside)
    return sigpix**2 * ompix * oneover_W_ell

def n_baseline(u, nu, exp_dict):
    """nu in [MHz] ! depending on the exp_dict either does the simplified calculation, or uses full baseline distribution"""
    if "n(x)" in exp_dict.keys():
        # print "using n(x) file"
        u = np.atleast_1d(u)
        x, nx = np.loadtxt(exp_dict["n(x)"], unpack = True)
        utab=x*nu #conversion from x to u
        nb_utab=nx/nu**2
        nb_uinterp = interp1d(utab, nb_utab,
        kind='linear', bounds_error=False, fill_value=1./INF_NOISE)
        res = nb_uinterp(u)
        res[np.where(res == 0.)] = 1./ INF_NOISE
        return res

    Na = exp_dict["Ndish"]
    Dmax = exp_dict["Dmax"]
    Dmin = exp_dict["Dmin"]
    lam = c/ nu #[m]

    return Na * (Na-1) * lam**2 / ( 2 * np.pi * (Dmax**2 - Dmin**2))

def noise_P_interferometer(ktab, nu, exp_dict):
    lam = c/ nu / 1e6 #[m]
    z = nutoz21(nu) #takes [MHz]
    r = rCom(z) #[Mpc]
    t_system = exp_dict["Tsys"] #[whatever]
    Sarea = exp_dict["S_area"] #[sterrad]
    time_tot = exp_dict["t_int"] * 60**2 #[s]

    y = c / H_z(z) * (1+z)**2 / (nu21emit * 1e6) #[Mpc s]
    theta = beam_FWHM(exp_dict, nu) #[rad]
    u = r * ktab / (2*np.pi)
    Ae = (exp_dict["Ddish"]/2)**2 * np.pi #collecting area [m2]
    num = (lam*mtoMpc)**4 * r**2 * y * t_system**2 * Sarea #[Mpc7 s mK sterrad]
    denom = 2 * (Ae*mtoMpc**2)**2 * theta**2 * n_baseline(u, nu, exp_dict) * time_tot #[Mpc4 sterrad s]
    return num/denom #[Mpc3 mK]

def noise_P_interferometer2(ktab, nu, dnu, exp_dict): #copying the calculation in baofisher
    lam = c/ nu / 1e6 #[m]
    z = nutoz21(nu) #takes [MHz]
    r = rCom(z) #[Mpc]
    t_system = exp_dict["Tsys"] #[whatever]
    Sarea = exp_dict["S_area"] #[sterrad]
    time_tot = exp_dict["t_int"] * 60**2 #[s]

    y = c / H_z(z) * (1+z)**2 / (nu21emit * 1e6) #[Mpc s]
    theta = beam_FWHM(exp_dict, nu) #[rad]
    u = r * ktab / (2*np.pi)
    Ae = (exp_dict["Ddish"]/2)**2 * np.pi #collecting area [m2]
    num = (lam*mtoMpc)**4 * t_system**2 * Sarea #[Mpc7 s mK sterrad]
    denom = 2 * nu21emit*1e6 * (Ae*mtoMpc**2)**2 * theta**2 * n_baseline(u, nu, exp_dict) * time_tot #[Mpc4 sterrad s]

    exponent = -y**2 * (dnu/nu21emit)**2 / ( 16* np.log(2))
    Bpar = np.exp(exponent)
    return num/denom * Bpar #[Mpc3 mK]

def Cl_interferom_noise(ltable, zmin, zmax, exp_dict): #from mario's mail on May 29th 2018
    numin = ztonu21(zmax); numax = ztonu21(zmin);
    nu = (numin+numax)/2
    dnu = (numax-numin)
    u = ltable / 2 / np.pi
    lam = c / 1e6 / nu #[m]
    Tsys = exp_dict["Tsys"] #[mK]
    Ae = (exp_dict["Ddish"]/2)**2 * np.pi #[m2]
    Sarea = exp_dict["S_area"] #[sterrad]
    ttot = exp_dict["t_int"] * 60**2 #[s]
    beam = beam_FWHM(exp_dict, nu) #[rad]

    Np = Sarea / beam**2
    tp = ttot / Np#time per pointing [s]
    num = (lam**2 * Tsys)**2 #[m2 mK]
    denom = Ae**2 * 2 * dnu*1e6 * n_baseline(u, nu, exp_dict) * tp  #[m2]
    return num/denom #[mK]

# def sigmaT_uv(u, nu, dnu, exp_dict):
#     lam = c / 1e6 / nu #[m]
#     Tsys = exp_dict["Tsys"] #[mK]
#     Ae = (exp_dict["Ddish"]/2)**2 * np.pi #[m2]
#     Sarea = exp_dict["S_area"] #[sterrad]
#     ttot = exp_dict["t_int"] * 60**2 #[s]
#     beam = beam_FWHM(exp_dict, nu) #[rad]
#
#     Np = Sarea / beam**2
#     tp = ttot / Np#time per pointing [s]
#     num = lam**4 * Tsys**2 #[(m2 mK)2]
#     # du = np.sqrt(Ae / lam**2) #page 30 in late-time cosmology (appendix)
#     denom = nu21emit * 1e6 *Ae**2 * n_baseline(u, nu, exp_dict) * tp  #[m4]
#     return num/denom #[mK]


# def Cl_interferom_noise(ltable, zmin, zmax, exp_dict):
#     zmean = (zmin+zmax)/2
#     nu = ztonu21(zmean)
#     dnu = ztonu21(zmin) - ztonu21(zmax)
#     # utable = ltable / 2 / np.pi
#     utable = ltable / 2 / np.pi
#     return sigmaT_uv(utable, nu, dnu, exp_dict)


def Cl_interferom_noise_slow(ltable,zmin,zmax, exp_dict, Nint = 500):
    """arguments: ell array, zmin, zmax
    returns: Cl as array"""
    ztable = np.linspace(zmin, zmax, Nint)
    integrand=np.zeros([len(ztable),len(ltable)])
    for l in range(len(ltable)):
        integrand[:,l]=np.array([E_z(zzz)*(W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*noise_P_interferometer((ltable[l])/rCom(zzz), ztonu21(zzz), exp_dict) for zzz in ztable])
    result=H_0/c*np.trapz(integrand,ztable,axis=0)
    return result

rescale = 1 #rescaling factor of nofz
def nofz(zz, mstar): #returns  the rescaled nz_distribtion function from lf_photometric in order to match the nz_gold.txt galaxy number
    zz = np.atleast_1d(zz)
    ntab = np.array([nz_distribution(zzz, mstar, "all")[0] for zzz in zz])
    return rescale * ntab

####################################################################################
# NOW: DETERMINE THE VALUE FOR rescale:
dNfile = "LSST/nz_gold.txt" #per arcmin!!
ztab_file, dNdztab_file = np.loadtxt(dNfile, unpack = True)
dNdzinterp = interp1d(ztab_file, dNdztab_file, kind='linear', bounds_error=False)
zint = np.linspace(0.001, 3.9, 1000)

nztab_func = nofz(zint, 27) # max lum cut
nztab_file = dNdzinterp(zint)

#go to all sky
nztab_func *= 4*np.pi #in sterrad
nztab_file *= 60**2 * 41200 #in arcmin

#integrate to get total number of galaxies:
Nz_func = np.trapz(nztab_func, zint)
Nz_file = np.trapz(nztab_file, zint)
rescale = Nz_file/Nz_func
print "\n", "#$"*30, "\n", "Rescaling the galaxy number density by a factor of {} to match the gold sample with {} total galaxies".format(rescale, Nz_file), "\n", "#$"*30, "\n"



def shotnoise(z, dz, galsurv, MAXMAG = False, NINT = 2000):
    # zmin = z - dz/2
    # zmax = z + dz/2

    zmin = z - dz #this used to be wrong, we define dz as half of the width...
    zmax = z + dz
    z_integrate = np.linspace(zmin,zmax, NINT)

    if type(MAXMAG) == bool and MAXMAG == False: #we use the maximum achievable magnitude cut and the stored table for LSST for the galaxy density
        ztab, dNdztab = np.loadtxt(galsurv["dNdz"], unpack = True)
        dNdzinterp = interp1d(ztab, dNdztab, kind='linear', bounds_error=False)
        dNzdOm = np.trapz(dNdzinterp(z_integrate), z_integrate)
        Nz = dNzdOm / (np.pi/180)**2 * 4 * np.pi #changing degrees to rad, and multiplying with all sky

    else: #we use the function for n(z) from lf_photometric from david's paper
        print "shot noise for LSST!"
        # dNdztabnew = np.array([nz_distribution(zzz, MAXMAG, "all")[0] for zzz in z_integrate])
        # dNdztabnew = dndz_fit(z_integrate, MAXMAG) #use the fitting function in the future!
        dNdztabnew = nofz(z_integrate, MAXMAG)
        dNzdOm = np.trapz(dNdztabnew, z_integrate)
        Nz = dNzdOm * 4 * np.pi #already in rad!!!! and multiplying with all sky
    return 4*pi/Nz #shot noise! The 4 pi could be cancelled but this way I can check more easily that it's correct...


####################################################################################################
####################################################################################################
####################################################################################################
#stuff to fit dndz for quicker calculations:

#power law times exponential to fit dndz (for speed!)
def powexp(x, a, zstar, alpha, beta):
    res = a * x**alpha * np.exp( -(x / zstar)**beta)
    return res
powexp_data_nam = "powexp_data/paramtab_for_powexp_in_magmod.txt"

paramnames = ["mstar", "a", "zstar", "alpha", "beta"]
powexp_readtab = ii.read(powexp_data_nam)

a0_inter = interp1d(powexp_readtab[paramnames[0]], powexp_readtab[paramnames[1]], kind = 'cubic', bounds_error = False)
zstar_inter = interp1d(powexp_readtab[paramnames[0]], powexp_readtab[paramnames[2]], kind = 'cubic', bounds_error = False)
alpha_inter = interp1d(powexp_readtab[paramnames[0]], powexp_readtab[paramnames[3]], kind = 'cubic', bounds_error = False)
beta_inter = interp1d(powexp_readtab[paramnames[0]], powexp_readtab[paramnames[4]], kind = 'cubic', bounds_error = False)

def powexp_allparams(mstar):
    return a0_inter(mstar), zstar_inter(mstar), alpha_inter(mstar), beta_inter(mstar)

def dndz_fit(z, mstar): # fits nofz! mstar must be float, not array
    aa, zzstar, alal, betbet = powexp_allparams(mstar)
    return powexp(z, aa, zzstar, alal, betbet)

def dndz_norm(zmin, zmax, mstar, FIT = True):
    """gets the normalization such that dndz integrates to one between zmin and zmax"""
    ztab = np.linspace(zmin,zmax, 1000)
    if FIT:
        dnndz = dndz_fit(ztab, mstar)
    else:
        dnndz = nofz(ztab, mstar)
    return np.trapz(dnndz, ztab)

def W_dndz(z, zmin, zmax, mstar, FIT = False):
    z = np.atleast_1d(z)
    if len(z)<5: #this is too small for normalization!
        raise ValueError("W_dndz needs to get vector redshift with length more than 5")
    if FIT:
        dnndz = dndz_fit(z, mstar)
    else:
        dnndz = nofz(z, mstar)
    if (zmin > z).any() or (zmax < z).any(): #some of our redshifts lie outside the bin
        cond = (z<zmin) | (z>zmax) #condition to be outside the [zmin, zmax] interval
        dnndz[cond] = 0 #we want to discard everything outside the bin!
    #normalize:
    if z[-1]<z[0]: #in this case z is an array running from big to small, giving problems with normalization
        return dnndz / np.trapz(dnndz[::-1], z[::-1])
    return dnndz / np.trapz(dnndz, z)


#two functions to get redshift bins (just to clean up the code:

def get_bg_bin_for_fgzmax(fzmax, bufferz = 0.1, bzmax = 3.9):
    # bzmax = zmax_of_MAXMAG(mstar)
    #bzmax = 3.9 for LSST. Note the end of the interpolation range of sgng at 3.9
    bzmin = fzmax + bufferz #minimum possible minimum redshift

    #now look for the place where the sign of 5sg-2 flips:
    #we do this to improve for SKA b2, which can't be optimized for mstar within the range [23,27]
    # zztab = np.linspace(0.1, 1., 1000)
    # sg5m2 = sg5minus2(zztab, 23) #because this is the lowest mstar we have implemented
    # flipi = np.where(sg5m2<0)[0][-1]+1 #+1 because we want first index after sign flip
    # bzmin2 = zztab[np.int(flipi)]
    # bzmin = np.amax([bzmin, bzmin2])


    zb = (bzmin +bzmax)/2
    dzb = (bzmax-bzmin)/2
    return bzmin, bzmax, zb, dzb



def get_fg_bin_for_frequency_range(nutot, numax):
    numin = numax - nutot
    nu = (numin + numax)/2
    zmax = nutoz21(numin)
    zmin = nutoz21(numax)
    zmin = np.amax((zmin, 1e-3))#zmin shouldn't be too small for the integration...
    dz = (zmax-zmin)/2
    z = (zmin+zmax)/2
    return zmin, zmax, z, dz

#number of galaxies behind redshift:
def ngal_behind(MAXMAG, NNUM = 100):
    ztab = np.linspace(1.5,4.,NNUM)
    # nztab = [nz_distribution(zz, MAXMAG, "all")[0] for zz in ztab]
    nztab = nofz(ztab, MAXMAG)
    ngal_before =integrate.cumtrapz( nztab, ztab, initial = 0  ) #all galaxies before z
    allgal = ngal_before[-1] #all galaxies
    ng_behind = allgal - ngal_before #all galaxies behind
    ng_behind *= 4 * np.pi # full sky
    return ng_behind, ztab

#towards finding a function zmax(m*), which is z until the last galaxy...:
def zmax_of_MAXMAG(MAXMAG, NNUM = 100):
    #calculates the maximum reach of LSST with a given magnitude cutoff.
    #I defined it to be the redshift where there is less than 1e-3 galaxies behind it on the full sky
    #NNUM = 1000 is tested to be accurate, but NNUM = 100 is probably enough as this is only a rough cutoff
    ng_behind,ztab = ngal_behind(MAXMAG, NNUM = NNUM)
    i_end = np.where(ng_behind < 1e-1)[0][0]
    return ztab[i_end]
#     return 2


#to calculate the S2N more easily, using mstar and fg redshift range.
def S2N_of_mstar_and_zf(mmag, ltabbb, zfmin, zfmax, bufferz = 0.1, SURVEY = [SKA, LSST],
                        P_FUNC_LIST = [C_l_HIHI_CAMB, C_l_gg_CAMB, Cl_HIxmag_CAMB]):
    # bzzmax = zmax_of_MAXMAG(mmag)
    zzf = (zfmin+zfmax)/2
    dzzf = (zfmax-zfmin)/2
    bzzmin, bzzmax, bzz, bdzz = get_bg_bin_for_fgzmax(zfmax, bufferz = bufferz)
    powspeclisttt = [P_FUNC_LIST[0](ltabbb, zfmin, zfmax),
                            P_FUNC_LIST[1](ltabbb, bzzmin, bzzmax, mmag),
                            P_FUNC_LIST[2](ltabbb, zzf, dzzf, bzz, bdzz, MAXMAG = mmag)]
    res = S2N(ltabbb, zzf, dzzf, bzz, bdzz, powspeclisttt, SURVEY=SURVEY, MAXMAG = mmag)
    return res

def S2N_for_opt(mmag, ltabbb, zfmin, zfmax, bufferz, SURVEY, P_FUNC_LIST):
    return -S2N_of_mstar_and_zf(mmag, ltabbb, zfmin, zfmax, bufferz = 0.1, SURVEY = [SKA, LSST],P_FUNC_LIST = [C_l_HIHI_CAMB, C_l_gg_CAMB, Cl_HIxmag_CAMB])[0]
