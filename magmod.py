from __future__ import division
import os
import numpy as np
import healpy as hp
from numpy import pi,sin,cos,tan,e,arctan,arcsin,arccos,sqrt
import sys
from scipy import integrate
import scipy.special as sp
from scipy.interpolate import interp2d, interp1d
from time import clock #timing


from homogen import hom, cosmo as cosmologies
from magbias_experiments import SKA_zhangpen, CLAR_zhangpen, SKA, hirax, n
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



############################################################################
############################################################################
#functions for the cross correlation power spectra
#
############################################################################
############################################################################


#preliminary, to be changed!!
bHI = 1
bgal = 1

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
def g(z, zb, dzb, NINT = 10000): #new standard to use cumtrapz
    """compare to my (handwritten) notebook 2 entry for 9/3/17
    zb is background redshift"""
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


def Cl_HIxmag_CAMB(ltable, zf, delta_zf, zb, delta_zb, Nint = 500):
    """ltable, zf foreground redshift, zb background redshift,
    delta_zf foreground redshift widht, Nint integration steps"""

    #note there is no T_obs factor because we want to compare with galaxy case
    fac1 = 3/2 * bHI*(H_0/c)**2 * Omega_m #no square on (H_0/c) because it cancels out
    fac2 = bHI * (5*sg(zb) - 2)
    zmin = zf - delta_zf
    zmax = zf + delta_zf
    ztab = np.linspace(zmin, zmax, Nint) #do checks on this! should be 0 to inf

    integrand=np.zeros([len(ztab),len(ltable)])
    for il in range(len(ltable)):
        ell = ltable[il]
        pknltab = np.array([pknl(( ell)/rCom(zzz), zzz) for zzz in ztab])
        integrand[:,il] = (1+ztab) * W_tophat(ztab, zmin, zmax) * g(ztab, zb, delta_zb) \
        / rCom(ztab)**2 * pknltab

        #old and slow ways to calculate the same thing:
#             integrand[:,il]= np.array([(1+zzz) * W_tophat(zzz, zfmin, zfmax) * g(zzz, zb, dzb) \
#                                        / rCom(zzz)**2 * pknl(( ell)/rCom(zzz), zzz) for zzz in ztab])
#         integrand[:,il]= np.array([(1+zzz) * W_tophat(zzz, zfmin, zfmax) * g(zzz, zb, dzb) * rCom(zzz) * pknl(( ell)/rCom(zzz), zzz) for zzz in ztab]) #different factor of chi than in alkistis' notes (units are wrong there)
    result= fac1 * fac2 * np.trapz(integrand,ztab,axis=0)
    return result



def C_l_HIHI_CAMB(ltable,zmin,zmax, Nint = 500):
    """arguments: ell array, zmin, zmax
    returns: Cl as array"""
    ztable = np.linspace(zmin, zmax, Nint)
    integrand=np.zeros([len(ztable),len(ltable)])
    for l in range(len(ltable)):
        #for the moment we divide out T_obs to compare with the galaxy results
#         integrand[:,l]=E_z(ztable)*(W_tophat(ztable,zmin,zmax)/rCom(ztable))**2*pknl((ltable[l])/rCom(ztable), ztable)
        integrand[:,l]= np.array([E_z(zzz)*(W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
    result=bHI**2 * H_0/c*np.trapz(integrand,ztable,axis=0)
    return result

def C_l_gg_CAMB(ltable,zmin,zmax, Nint = 500):
    """arguments: ell array, zmin, zmax
    returns: Cl as array"""
    ztable = np.linspace(zmin, zmax, Nint)
    integrand=np.zeros([len(ztable),len(ltable)])
    for l in range(len(ltable)):
        integrand[:,l]=np.array([E_z(zzz)*(W_tophat(zzz,zmin,zmax)/rCom(zzz))**2*pknl((ltable[l])/rCom(zzz), zzz) for zzz in ztable])
    result=bgal**2 * H_0/c*np.trapz(integrand,ztable,axis=0)
    return result


def DELTA_Cl_HIxmag(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = "CV", nside = 256):
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
            Cshot = shotnoise(zb, dzb, SURVEY[1]);
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
    num = X2 + (HIHI + N_ell) * (gg + Cshot)
    denom = (2*ltable+1) * d_ell * fsky
    result = np.sqrt(num/denom)
    return result

def S2N(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = "CV"):
    start = clock()
    delt = DELTA_Cl_HIxmag(ltable, zf, dzf, zb, dzb, power_spectra_list, SURVEY = SURVEY)
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

def sg(z, experiment = CLAR_zhangpen, USE_ALPHA = False): #number count slope. Fit taken from eq 23 in 1611.01322v2
    """if USE_ALPHA, then sg is just caluclated from alpha"""
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

        return n0 + n1*z + n2*z**2 + n3*z**3 + n4*z**4 + n5*z**5





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



def G(z, zmin, zmax, nsig, experiment, NINT = 1000): #kernel of the bg magnification
    """zmin and zmax are edges of galaxy distribution we're looking at
    experiment needs to be SKA etc because of alpha"""
    z = np.atleast_1d(z)
    zdist = np.linspace(zmin, zmax, NINT)
    chi = rCom( z )
    chidist = rCom(zdist) #of galaxy distribution, etiher bg or fg
#     wmatrix = w_geometry(chi, chidist)
    ntab = dNdz(zdist, nsig, experiment)
    Ngal = np.trapz(ntab, zdist)
    alphatab = alpha(zdist, experiment)
#     alphatab = 5/2*sg(zdist) #same thing
    integrand1 = ntab * 2 * (alphatab -1 )
    integrand2 = ntab * 2 * (alphatab -1 ) / chidist
    integral1 = np.trapz( integrand1, zdist)
    integral2 = np.trapz( integrand2, zdist)
    return ((1+z)/Ngal) * (chi * integral1 - chi**2 * integral2)

def Cl_gxmag_CAMB(ltable, zf, delta_zf, zb, delta_zb, nsig, experiment, Nint = 500):
    zftab = np.linspace(zf-delta_zf, zf+delta_zf, Nint)
    zbmin = zb - delta_zb
    zbmax = zb + delta_zb
    rtab = rCom(zftab)
    ntab = dNdz(zftab, nsig, experiment)
    Ngal = np.trapz(ntab, zftab)
    integrand=np.zeros((len(zftab),len(ltable)))
    for il in range(len(ltable)):
        ell = ltable[il]
        pknltab = np.array([pknl(( ell)/rCom(zzz), zzz) for zzz in zftab])
        #we divide by r**3 because we go from Delta_m to P_m
        integrand[:,il]= 1/rtab**3 * pknltab * G(zftab, zbmin, zbmax, nsig, experiment) * dNdz(zftab, nsig, experiment) * rtab
    fac1 = 3 * Omega_m * H_0**2  / c**2
    fac2 = np.pi**2#/ ltable**3 #we use k^3 Delta_m = l^3/r^3 Delta_m = P_m
    fac3 = 1 / Ngal
    result= fac1 * fac2 * fac3 * np.trapz(integrand,zftab,axis=0)
    return result * h**6



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

def shotnoise(z, dz, galsurv, NINT = 200):
    print "We use all sky for calculating N(z)"
    zmin = z - dz/2
    zmax = z + dz/2
    ztab, dNdztab = np.loadtxt(galsurv["dNdz"], unpack = True)
    dNdzinterp = interp1d(ztab, dNdztab, kind='linear', bounds_error=False)
    z_integrate = np.linspace(zmin,zmax, NINT)
    dNzdOm = np.trapz(dNdzinterp(z_integrate), z_integrate)
    Nz = dNzdOm / (np.pi/180)**2 * 4 * np.pi #changing degrees to rad, and multiplying with survey area
    return 4*pi/Nz #shot noise! The 4 pi could be cancelled but this way I can check more easily that it's correct...
