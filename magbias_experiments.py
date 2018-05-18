import numpy as np

#this is where we store the experiment parameters for noise:

n = 2 #n is the detection threshold (in #sigma)
FOV_SKA = 1 #[deg2]
A_SKA = 25600 #[deg2]

FOV_CLAR = 1 #[deg2]
# A_CLAR = 640 #[deg2]
A_CLAR = 160 #[deg2]
# A_CLAR = 40 #[deg2]
# A_CLAR = 10 #[deg2]

ttot = 5 * 365 * 24 #[hours], 5 years



SKA_zhangpen = { "Aeff" : 6e5, #[m2]
      "Tsys" : 30, #[K]
      "t_int" : ttot * FOV_SKA / A_SKA,
       "S_area": A_SKA,
       "Name" : "SKA_zhangpen"
      }

CLAR_zhangpen = {
    "Aeff" : 5e4, #[m2]
    "Tsys" : 30, #[K]
    "t_int": ttot * FOV_CLAR / A_CLAR,
    "S_area": A_CLAR,
    "Name" : "CLAR_zhangpen"
}

SKA = {
    "mode":             "single_dish",
    "Ndish":            133+64,
    "Tsys":             30 * 1e3, #mK
    "S_area":           7.6,    #25000 sq deg = 7.6 [sterrad]
    "t_int":            8760,   #1 yr in [h], for comparison with hirax
    "Ddish":            (133 * 15 + 64 * 13.5) / (133+64), #average [m]
    "Nbeam":            1,
    "Name" :            "SKA",
    }


hirax = {
    "mode":             'interferometer',
    'Ndish':            1024,                 # No. of dishes
    'Nbeam':            1,                # No. of beams (for multi-pixel detectors)
    'Ddish':            6.,               # Single dish diameter [m]
    "t_int":            8760,               #1 year in [h]
    'Tsys':            50.*(1e3),         # System temp. [mK]
    #'nu_crit':          1000.,            #critical frequency, UNCLEAR
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,             # Max. freq. of survey
    'dnu':              0.4,               # Bandwidth of single channel [MHz]
    'S_area':            np.pi,          # fsky = 1/4 Total survey area [radians^2]
    'Dmax':             307.,               # Max. interferom. baseline [m]
    'Dmin':             7.,                 # Min. interferom. baseline [m]
}
cb_hirax=dict(hirax) #same but with actual baseline file.
cb_hirax.update({'n(x)':             "/Users/amadeus/Documents/PhD/work/bao21cm/hirax/hirax_Ndish1024_baseline1.dat"})

LSST = {
    "dNdz":         "/Users/amadeus/Documents/PhD/work/multitracer/LSST/nz_LSST_gold_sqdeg.dat",
    }
