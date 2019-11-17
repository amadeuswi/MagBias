from __future__ import division
import cosmology as cosmo
import numpy as np
from scipy.integrate import odeint



#unit conversions
Mpctom=3.0857*10**22
mtoMpc=1/Mpctom
sectoyears=1/60/60/24/365
c=3e8 #m/s

#cosm=cosmo.std
#omega_k=cosm['omega_k']
#omega_m=cosm['omega_M_0']
#omega_lambda=cosm['omega_lambda_0']

#defining H_0 in m/s/Mpc
#H_0=100*cosm['h']
#H_0*=1000
#print "*"*10+"homogeneous.py: H_0 units: m/s/Mpc"+"-"*10


def set_cosmology(cosmostring="standard"):
    """input: string specifying the cosmology in the cosmology.py dict, "standard" for Planck
    output: none, sets global variables"""
    global cosm,omega_k,omega_m,omega_lambda,H_0,H_0_per_second,h,omega_b,sigma_8,n_s,Tcmb
    if type(cosmostring)==dict:
        cosm=cosmostring #cosmostring in this case is actually a dict
    else:
        if cosmostring == "standard":
            cosm=cosmo.std
        else:
            cosm=cosmo.cosmologies[cosmostring]
    sigma_8=cosm['sigma_8']
    omega_k=cosm['omega_k']
    omega_b=cosm['omega_b_0']
    omega_m=cosm['omega_M_0']
    omega_lambda=cosm['omega_lambda_0']
    if cosmostring == "standard":
        global omega_r,a_eq
        omega_r=cosm['omega_r_0']
        a_eq=omega_r/omega_m
    else:
        print("*"*50)
        print("a_eq and omega_r not set!")
        print("*"*50)
    #defining H_0 in m/s/Mpc:
    h=cosm['h']
    H_0=100*cosm['h']
    H_0*=1000
    n_s=cosm['ns']
    Tcmb=cosm['Tcmb']
    #defining H_0 in 1/s:
    H_0_per_second=H_0*mtoMpc

    #check behaviour:
#    if omega_m**2-4*omega_k*omega_lambda<10**-3:
#        print "There is a maximal scale factor in cosmology %s" %(cosmostring)
#    else:
#        print "There is probably no maximal scale factor in cosmology %s" %(cosmostring)
#
set_cosmology()


def frequency_obs_of_z(z, f_emit):
    return f_emit/(1+z)
def z_of_frequency_obs(f_obs, f_emit):
    return f_emit/f_obs - 1


def zofa(a):
    return 1/a-1

def aofz(z):
    return 1/(z+1)


def E_z(z,cosmo=False):
#    if z<0:
#        raise ValueError("negative redshift")
    """pivot scale not implemented!"""
    if cosmo==False:
        arg=(omega_m)*(1+z)**3+omega_k*(1+z)**2+omega_lambda
    else:
        if 'domega_lambda_by_dz' in cosmo.keys():  dol_by_dz = cosmo['domega_lambda_by_dz'] #parameter for z dependence of omega_lambda. introduced to add a constant DE term to equation (10) in Knox (arxiv:0503405v2)
        else:   dol_by_dz = 0
        arg=(cosmo['omega_M_0'])*(1+z)**3+cosmo['omega_k']*(1+z)**2+cosmo['omega_lambda_0']+dol_by_dz*z
    if np.amin(arg)<.1:
        arg=np.abs(arg)

    return (arg)**0.5

def set_apiv():
    """get pivot scale a where the sign of E changes"""
    global apiv, Epiv
    aa=np.linspace(0.1,3,30000)
    zz=zofa(aa)
    Evec=E_z(zz)
    Eprime=(Evec[1:]-Evec[:-1])/(zz[1:]-zz[:-1])#one length lost
    Eprime_diff=(Eprime[1:]-Eprime[:-1])#two length lost
    Eprime_ddiff=np.abs(Eprime_diff[1:]-Eprime_diff[:-1])#three length lost
    Eprime_ddiff/=np.amax(Eprime_ddiff)
    cond0=Eprime_ddiff[:-5]>.5
    cond1=Eprime_ddiff[5:]<.1
    cond2=Eprime_ddiff[1:-4]<(.4*Eprime_ddiff[:-5])
    cond3=aa[4:-4]>.3
    maxind=cond0*cond1*cond2*cond3
    apiv=aa[maxind]#plus two because length lost in finding derivative
    if len(np.where(maxind==True)[0])>1:
        print("set_apiv: more than one pivot a!")
    Epiv=Evec[maxind]
    if len(apiv)<1:
        apiv='None'
        Epiv='None'
#set_apiv()

def E_a(a,cosmo=False):
    return E_z(zofa(a),cosmo=cosmo)


def get_Evec(aaavec,cosmo=False,universe_starts_contracting=False):
    """takes vector a input, returnes vector E which changes sign at global apiv"""
    avec=np.copy(aaavec)
    Evec=E_a(avec,cosmo=cosmo)
    if apiv=='None':
        return Evec,avec

    Evec[avec>apiv]*=-1
    avec[avec>apiv]=2*apiv-avec[avec>apiv]
    avec[avec<0.]=0
    if universe_starts_contracting:
        Evec*=-1
    return Evec,avec

def H_z(z,cosmo=False):
    """[m/s/Mpc]"""
    if cosmo == False:  H0=H_0
    else:   H0=cosmo['H_0']
    return H0*E_z(z,cosmo=cosmo)

def H_a(a,cosmo=False):
    if cosmo == False:  H0=H_0
    else:   H0=cosmo['H_0']
    return H0*E_a(a,cosmo=cosmo)

def timeintegrand(a,years=False):
    result = Mpctom/(a*H_a(a))
    if years: result*=sectoyears
    return result

def time_maxscale(avec,aaaend):
    aend=np.copy(np.atleast_1d(aaaend))
    if apiv=='None':
        print("no maximal scale factor")
        return('None')
    newaend=aend[aend<apiv]
    Evec1=E_a(newaend)
    argument=1/newaend/H_0_per_second/Evec1
    res=[np.trapz(argument[:end],newaend[:end]) for end in np.arange(len(newaend))]
    res=np.array(res)
    res2=-res[::-1]+2*res[-1]
    newaend2=newaend[::-1]
    return np.append(res,res2),np.append(newaend,newaend2)



def time(aaa,return_a=True):
    """input: cosmic scale factor a (scalar or np.ndarray), also bool wether years or seconds should be used, and wether a scalar (in that case only first entry of vector) or a list should be returned.
    output: cosmic time!"""
    if apiv!='None':
        return time_maxscale(aaa,aaa)
    a=np.atleast_1d(np.copy(aaa))
    Evec=E_a(a)
    argument=1/a/H_0_per_second/Evec
    res=[np.trapz(argument[:end+1],a[:end+1]) for end in np.arange(len(a)-1)]
    res=np.array(res)
    if return_a:
        return res,a[:-1]
    else:
        return res
#    res_err=[integrate.quad(timeintegrand,astart,aa,args=(years,))
#               for aa in a]
#    result=[res_err[i][0] for i in range(len(a))]
#    error=[res_err[i][1] for i in range(len(a))]
#    if not return_list or len(result)==1:
#        result=result[0]
#        error=error[0]
#    if return_error:
#        return result,error
#    else:
#        return result

#def scalefactor_DE_arg(a,y):
#    return H_a(a)*a
#
#def scalefactor_DE(tvec):
#    init = [0,0]
#    sol=integrate.odeint(scalefactor_DE_arg,init,tvec)
#    return sol


def dynamical_q(ll,kk,w=0):
    gamma=w+1
#    l=np.outer(ll,1)
    return 1/2*((3*gamma-2)*(1-kk)-3*gamma*ll)

def dynamical_lprime(ll,kk,w=0):
    return 2*(1+dynamical_q(ll,kk,w=w))*ll
def dynamical_kprime(ll,kk,w=0):
    return 2*dynamical_q(ll,kk,w=w)*kk
def dynamical_lkprime(y,t=0,w=0):
    ll=y[0]
    kk=y[1]
    lprime=dynamical_lprime(ll,kk,w=w)
    kprime=dynamical_kprime(ll,kk,w=w)
    return np.array([lprime,kprime])

def dynamical_increment(initllkk,w=0,epsilon=0.001):
    gradient=dynamical_lkprime(initllkk,w=w)
    new=initllkk+epsilon*gradient
    return new

def dynamical_line(initllkk,steps,epsilon,w=0):
    lk=[]
    ll=[]
    kk=[]
    lk.append(initllkk)
    ll.append(initllkk[0])
    kk.append(initllkk[1])
    for step in np.arange(steps):
        new=dynamical_increment(lk[-1],epsilon=epsilon,w=w)
        lk.append(new)
        ll.append(new[0])
        kk.append(new[1])
    return lk,ll,kk

def rphi_to_xy(r,phi):
    x=r*np.cos(phi)
    y=r*np.sin(phi)
    return x,y

def dynamical(initllkk,w=0):
    initlk=np.copy(initllkk)
    solution=odeint(dynamical_lkprime,initlk,np.linspace(-100,0))
    llvec=solution[:,0]
    kkvec=solution[:,1]
    return llvec,kkvec

def CHI_a(a,cosmo=False):#comov distance to object at scale factor a, eq 2.42 in Dodelson, [Mpc]
    atab=np.linspace(a,1,200)
    Htab=H_a(atab,cosmo=cosmo)#units: m/s/Mpc
    integrand=c/atab**2/Htab#units: Mpc
    result=np.trapz(integrand,atab)
    return result #units: Mpc

def CHI_z(z,cosmo=False):
    return CHI_a(aofz(z),cosmo=cosmo)


def dA_flat_z(z):
    """only flat cosmologies, returns Mpc"""
    z1d=np.atleast_1d(z)
    result=[CHI_z(zz)/(1+zz) for zz in z1d]
    return np.array(result)
#set_cosmology()
