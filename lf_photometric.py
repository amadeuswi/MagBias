#comes from http://intensitymapping.physics.ox.ac.uk


import numpy as np
import matplotlib.pyplot as plt

# import py_cosmo_mad as pcs
from scipy.integrate import quad
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import scipy.special as sp

import copy as copy
from homogen import hom
###########################
# Cosmological parameters #
###########################




hhub=0.7
h = hhub
omm=0.3
Omega_m = omm
oll=0.7
Omega_lambda = oll

H_z = hom.H_z

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

def schechter(m,phs,ms,al) :
    """Schechter function."""
    tmm=10**(0.4*(ms-m))
    return 0.4*np.log(10.)*phs*tmm**(al+1)*np.exp(-tmm)


#######################
# LF for red galaxies #
#######################
bmr=1.32
alpha_red=-0.5
def mb_star_red(z) :
    """M_star in the Schechter function."""
    return -20.6233-0.490061*z

def phib_red_star(z) :
    """phi_star in the Schechter function (in units of 10^-3 Mpc^-3)."""
    return 0.00181758/(1+(z/1.04456)**7.16891)


def lm_red_b(mb,z) :
    """LF for red galaxies in the B-band."""
    mbs=mb_star_red(z)
    phs=phib_red_star(z)/hhub**3

    return schechter(mb,phs,mbs,alpha_red)

def lm_red_r(mr,z) :
    """LF for red galaxies in the r-band."""
    mb=mr+bmr

    return lm_red_b(mb,z)


#######################
# LF for all galaxies #
#######################
alpha_all=-1.33
m0s_all=-21.49
a_all=-1.25
ph0s_all=0.0042
b_all=-0.85
def mr_star_all(z) :
    """M_star in the Schechter function."""
    return m0s_all+a_all*np.log(1+z)

def phr_star_all(z) :
    """phi_star in the Schechter function."""
    z = np.atleast_1d(z)
    phr=-8.12065e-05*z**2-0.000136239*z+0.00258397
    phr[np.where(phr <= 0)] = 0
    return phr
    # if phr>0 :
    #     return phr
    # else :
    #     return 0

def lm_all_r(mr,z) :
    """LF for all galaxies in the r-band."""
    phs=phr_star_all(z)/hhub**3
    mrs=mr_star_all(z)

    return schechter(mr,phs,mrs,alpha_all)


########################
# LF for blue galaxies #
########################
def lm_blue_r(mr,z) :
    """LF for blue galaxies in the r-band."""
    z = np.atleast_1d(z)
    lred=lm_red_r(mr,z)
    lall=lm_all_r(mr,z)

    # return lall-lred
    ldiff = lall-lred
    ldiff_neg = ldiff<=0
    if ldiff_neg.any:
        ldiff[ldiff_neg]=0
    if len(ldiff) == 1:
        return ldiff[0]
    return ldiff


###################################
# Apparent to absolute magnitudes #
###################################
def app2abs(r,z,typ) :
    """Transform from apparent to absolute magnitudes."""
    # chi=pcs.radial_comoving_distance(1./(1+z))
    chi=rCom(z)
    dl=(1+z)*chi
    if typ=="red" :
        kcorr=2.5*z
    elif typ=="blue" :
        kcorr=1.5*z

    return r-5*np.log10(dl)-25+np.log10(hhub)-kcorr


#######################
# Luminosity function #
#######################
def lumfun_mag(mr,z,typ) :
    """Wrapper for the luminosity function."""
    if typ=="red" :
        return lm_red_r(mr,z)
    elif typ=="blue" :
        return lm_blue_r(mr,z)
    elif typ=="all" :
        return lm_all_r(mr,z)


def lumfun(lnlum,z,typ) :
    """LF as a function of ln(L_r)"""
    if type(z) != float and type(z) != np.float64:
        print("z is wrong: {}".format(type(z)))
        raise ValueError
    if z==0 :
        return 0
    else :
        ilten=1./np.log(10.)
        norm=2.5*ilten
        mr=-norm*lnlum
        return norm*lumfun_mag(mr,z,typ)

def cumulative_lumfun(mag_lim_red,mag_lim_blue,z,typ) :
    """Cumulative luminosity function."""
    z = np.atleast_1d(z)
    mag_lim_red = np.atleast_1d(mag_lim_red)
    mag_lim_blue = np.atleast_1d(mag_lim_blue)
    norm=-0.4*np.log(10.)
    lnlum_max=-35.*norm

    if typ=="red" :
        # return quad(lumfun,norm*mag_lim_red,lnlum_max,args=(z,typ))[0]
        clred = [quad(lumfun,norm*mag_lim_red,lnlum_max,args=(z[i],typ))[0] for i in range(len(z))]#amadeus' edit
        return np.array(clred)

    elif typ=="blue" :
        # return quad(lumfun,norm*mag_lim_blue,lnlum_max,args=(z,typ))[0]
        clblue =  [quad(lumfun,norm*mag_lim_blue,lnlum_max,args=(z[i],typ))[0] for i in range(len(z))]#amadeus' edit
        return np.array(clblue)
    elif typ=="all" :
        # clred=[quad(lumfun,norm*mag_lim_red,lnlum_max,args=(zzz,"red"))[0] for zzz in z]
        clred=[quad(lumfun,norm*mag_lim_red[i],lnlum_max,args=(z[i],"red"))[0] for i in range(len(z))]
        clblue=[quad(lumfun,norm*mag_lim_blue[i],lnlum_max,args=(z[i],"blue"))[0] for i in range(len(z))]
        return np.array(clred)+np.array(clblue)



#by Amadeus: testing the cumulative lumfun
def cumulative_lumfun_all(mag_lim,z,typ) :
    """Cumulative luminosity function."""
    z = np.atleast_1d(z)
    mag_lim = np.atleast_1d(mag_lim)
    norm=-0.4*np.log(10.)
    lnlum_max=-35.*norm

    if typ!="all" :
        raise ValueError("this function only works for type all!")

    clall=[quad(lumfun,norm*mag_lim[i],lnlum_max,args=(z[i],"all"))[0] for i in range(len(z))]
    return np.array(clall)


########
# N(z) #
########
def nz_distribution(z,rmax,typ) :
    """Angular number density (per srad) as a function of redshift."""
    if z==0.0 :
        z=0.001
    maglim_red=app2abs(rmax,z,"red")
    maglim_blue=app2abs(rmax,z,"blue")

    chi=rCom(z)
    ih=1./(H_z(z)/1000)

    lumPhi=cumulative_lumfun(maglim_red,maglim_blue,z,typ)

    return lumPhi*chi**2*ih


########
# s(z) #
########
def s_magbias_original(z,rmax,typ) :
    """Magnification bias."""
    # z = np.atleast_1d(z)
    if z==0.0 :
        z=0.001

    ilten=1./np.log(10.)
    norm=2.5*ilten
    maglim_red=app2abs(rmax,z,"red")
    maglim_blue=app2abs(rmax,z,"blue")
    lumPhi=cumulative_lumfun(maglim_red,maglim_blue,z,typ)
    if typ=="red" :
        lumphi=norm*lumfun_mag(maglim_red,z,typ)
    elif typ=="blue" :
        lumphi=norm*lumfun_mag(maglim_blue,z,typ)
    elif typ=="all" :
        lumphi_red=norm*lumfun_mag(maglim_blue,z,"red")
        lumphi_blue=norm*lumfun_mag(maglim_blue,z,"blue")
        lumphi=lumphi_blue+lumphi_red

    return 0.4*lumphi/lumPhi

def s_magbias(z,rmax,typ) : #trying to speed it up
    """Magnification bias."""
    z = np.atleast_1d(z)
    z[np.where(z==0.0)]+=0.001

    ilten=1./np.log(10.)
    norm=2.5*ilten
    maglim_red=app2abs(rmax,z,"red")
    maglim_blue=app2abs(rmax,z,"blue")
    lumPhi=cumulative_lumfun(maglim_red,maglim_blue,z,typ)
    if typ=="red" :
        lumphi=norm*lumfun_mag(maglim_red,z,typ)
    elif typ=="blue" :
        lumphi=norm*lumfun_mag(maglim_blue,z,typ)
    elif typ=="all" :
        lumphi_red=norm*lumfun_mag(maglim_blue,z,"red")
        lumphi_blue=norm*lumfun_mag(maglim_blue,z,"blue")
        lumphi=lumphi_blue+lumphi_red

    return 0.4*lumphi/lumPhi



#testing function from amadeus:
def s_magbias_doublereturn(z,rmax,typ) : #just like the original, but different return
    """Magnification bias."""
    z = np.atleast_1d(z)
    z[np.where(z==0.0)]+=0.001

    ilten=1./np.log(10.)
    norm=2.5*ilten
    maglim_red=app2abs(rmax,z,"red")
    maglim_blue=app2abs(rmax,z,"blue")
    lumPhi=cumulative_lumfun(maglim_red,maglim_blue,z,typ)
    if typ=="red" :
        lumphi=norm*lumfun_mag(maglim_red,z,typ)
    elif typ=="blue" :
        lumphi=norm*lumfun_mag(maglim_blue,z,typ)
    elif typ=="all" :
        lumphi_red=norm*lumfun_mag(maglim_blue,z,"red")
        lumphi_blue=norm*lumfun_mag(maglim_blue,z,"blue")
        lumphi=lumphi_blue+lumphi_red

#this is the only line changed:
    return lumphi,lumPhi


############
# f_evo(z) #
############
dz_constant=0.01
def f_evo(z,rmax,typ) :
    if z==0.0 :
        z=0.001
    ilten=1./np.log(10.)
    norm=2.5*ilten
    maglim_red=app2abs(rmax,z,"red")
    maglim_blue=app2abs(rmax,z,"blue")
    plumPhi=np.log(cumulative_lumfun(maglim_red,maglim_blue,z+dz_constant,typ))
    if z>dz_constant :
        mlumPhi=np.log(cumulative_lumfun(maglim_red,maglim_blue,z-dz_constant,typ))
        dlumPhi=(plumPhi-mlumPhi)/(2*dz_constant)
    else :
        mlumPhi=np.log(cumulative_lumfun(maglim_red,maglim_blue,0.0001,typ))
        dlumPhi=(plumPhi-mlumPhi)/(z+dz_constant)

    return -(1+z)*dlumPhi

########
# Bias #
########
def bias(z,rmax,typ) :
    if z==0.0 :
        z=0.001
    if typ=="red" :
        return 2.0+1.0*(z-1.0)
    elif typ=="blue" :
        bz_red=bias(z,rmax,"red")
        bz_all=bias(z,rmax,"all")
        nz_red=nz_distribution(z,rmax,"red")
        nz_all=nz_distribution(z,rmax,"all")

        return (nz_all*bz_all-bz_red*nz_red)/(nz_all-nz_red)
    else :
        return 1+0.84*z


if __name__=="__main__":
    # pcs.background_set(omm,oll,0.049,-1,0,hhub,2.7255)
    rmax=27.
    # rmax=28.
    nzbins=384
    # zarr_red=2.0*(np.arange(nzbins)+0.0)/nzbins
    zarr_all=3.7*(np.arange(nzbins)+0.0)/nzbins
    rad2amin=(np.pi/180./60.)**2
    # rmax_gold=26.3
    fs=16

    nz_full_all_arr=np.array([nz_distribution(z,rmax,"all") for z in zarr_all])*rad2amin
    plt.xlim([0,3.7])
    plt.xlabel("$z$",fontsize=fs)
    plt.ylabel("$dN/(dz\,d\\Omega)\\,[{\\rm amin}^{-2}]$",fontsize=fs)
    plt.plot(zarr_all,nz_full_all_arr,'r-',label='all')
    plt.legend(loc='upper right')
    plt.show()

    bz_full_all_arr=np.array([bias(z,rmax,"all") for z in zarr_all])
    # plt.xlim([0,3.7])
    plt.xlim([0,1.5])
    plt.xlabel("$z$",fontsize=fs)
    plt.ylabel("$b(z)$",fontsize=fs)
    plt.plot(zarr_all,bz_full_all_arr,'r-',label='all')
    plt.legend(loc='upper right')
    plt.show()

    sz_full_all_arr=np.array([s_magbias(z,rmax,"all") for z in zarr_all])
    # plt.xlim([0,2.0])
    plt.xlim([0,1.5])
    # plt.ylim([-1,8])
    plt.ylim([0,0.5])
    plt.xlabel("$z$",fontsize=fs)
    plt.ylabel("$s(z)$",fontsize=fs)
    plt.plot(zarr_all,sz_full_all_arr,'r-',label='all')
    plt.legend(loc='upper right')
    plt.show()

    ez_full_all_arr=np.array([f_evo(z,rmax,"all") for z in zarr_all])
    # plt.xlim([0,3.7])
    plt.xlim([0,1.5])
    # plt.ylim([-2,0])
    plt.ylim([-1,0])
    plt.xlabel("$z$",fontsize=fs)
    plt.ylabel("$f_{\\rm evo}(z)$",fontsize=fs)
    plt.plot(zarr_all,ez_full_all_arr,'r-',label='all')
    plt.legend(loc='upper right')
    plt.show()
