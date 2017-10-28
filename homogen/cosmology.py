# Define fiducial cosmology and parameters
# Planck-only best-fit parameters, from Table 2 of Planck 2013 XVI.
import numpy as np


zstar = 1090.09
basic = { #parameters shared by all cosmologies
    'Tcmb':             2.7255#CMB temperature in Kelvin
    }

std = {
    'omega_M_0':        0.316,
    'omega_lambda_0':   0.684,
    'omega_b_0':        0.049,
    'omega_HI_0':       4.86e-4, #6.50e-4,
    'omega_k':          0.0,
    'omega_r_0':        9.1263e-05, # follows from CMB temperature of 2.725K + neutrinos
    'N_eff':            3.046,
    'h':                0.67,
    'H_0':              0.67*1e5,
    'ns':               0.962,
    'sigma_8':          0.834,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'k_piv':            0.05, # n_s
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.677, #0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'b_1':              0.,         # Scale-dependent bias (k^2 term coeff.)
    'k0_bias':          0.1,        # Scale-dependent bias pivot scale [Mpc^-1]
    'A_xi':             0.,         # Modified gravity growth amplitude
    'k_mg':             1e-2        # New modified gravity growth scale
}
std.update(basic)

pwc = {
    'omega_M_0':        0.316,
    'omega_lambda_0':   0.684,
    'omega_b_0':        0.049,
    'omega_HI_0':       4.86e-4, #6.50e-4,
    'omega_k':          0.0,
    'w0':          -1.0,
    'w1':          -1.0,
    'omega_r_0':        9.2109e-05, # follows from CMB temperature of 2.725K + neutrinos
    'N_eff':            3.046,
    'h':                0.67,
    'H_0':              0.67*1e5,
    'ns':               0.962,
    'sigma_8':          0.834,
    'gamma':            0.55,
    'fNL':              0.,
    'mnu':              0.,
    'k_piv':            0.05, # n_s
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.677, #0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'b_1':              0.,         # Scale-dependent bias (k^2 term coeff.)
    'k0_bias':          0.1,        # Scale-dependent bias pivot scale [Mpc^-1]
    'A_xi':             0.,         # Modified gravity growth amplitude
    'k_mg':             1e-2,        # New modified gravity growth scale
    'zstar':            zstar,
    'w_DE_zvec':        [  0.00000000e+00, 1.09109000e+03],
}

w0wa = dict(pwc)
del w0wa['w1'];
w0wa.update({'wa': 0.0, 'w_DE_zvec': False})


#several non-stdt cosmologies
cosmologies = {
    'A':        {'omega_M_0':1.0,'omega_lambda_0':0.0,'omega_k':0.0,'h':0.67},
    'D':        {'omega_M_0':3.0,'omega_lambda_0':0.0,'omega_k':-2.0,'h':0.67},
    'H':        {'omega_M_0':3.0,'omega_lambda_0':1.0,'omega_k':-3.0,'h':0.67},
    'F':        {'omega_M_0':0.0,'omega_lambda_0':1.0,'omega_k':0.0,'h':0.67},
    'N':        {'omega_M_0':0.1,'omega_lambda_0':2.5,'omega_k':-1.6,'h':0.67},
    'G':        {'omega_M_0':3.0,'omega_lambda_0':.1,'omega_k':-2.1,'h':0.67},
    'custom':   {'omega_M_0':0.0,'omega_lambda_0':2.0,'omega_k':-1.0,'h':0.67}
#    'Koda_Blake':   {
#                        'omega_M_0':        0.273,
#                        'omega_lambda_0':   0.727,
#                        'omega_b_0':        0.0546,
#                        'omega_k':          0.0,
#                        'h':                0.705,
#                        'sigma_8':          0.812,
#                        'ns':               0.961
#                    }#cosmology used in arXiv:1312.1022v1

}
cosmologies['Koda_Blake']=dict(std)
cosmologies['Koda_Blake'].update({
                        'omega_M_0':        0.273,
                        'omega_lambda_0':   0.727,
                        'omega_b_0':        0.0546,
                        'omega_k':          0.0,
                        'h':                0.705,
                        'sigma_8':          0.812,
                        'ns':               0.961
})


for key in cosmologies.keys():
    cosmologies[key].update(basic)
