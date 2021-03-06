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
      "Tsys" : 30*1e3, #[mK]
      "t_int" : ttot * FOV_SKA / A_SKA,
       "S_area": A_SKA,
       "Name" : "SKA_zhangpen"
      }

CLAR_zhangpen = {
    "Aeff" : 5e4, #[m2]
    "Tsys" : 30*1e3, #[mK]
    "t_int": ttot * FOV_CLAR / A_CLAR,
    "S_area": A_CLAR,
    "Name" : "CLAR_zhangpen"
}

SKA = {
    "mode":             "single_dish",
    "Ndish":            133+64,
    "Tsys":             30 * 1e3, #mK
    # "S_area":           7.6,    #25000 sq deg = 7.6 [sterrad]
    "S_area":           5.15,    #fsky of 0.41 as in the multitracer paper [sterrad]
    "t_int":            8760,   #1 yr in [h], for comparison with hirax
    "Ddish":            (133 * 15 + 64 * 13.5) / (133+64), #average [m]
    "Nbeam":            1,
    "Name" :            "SKA",
    }

#bands 1 and 2 are copies of SKA with different mean Tsys:
#do not forget that they have different but overlapping frequency coverage
#numin and numax taken from /Users/amadeus/Documents/PhD/work/multitracer/experiments/SKA_Tsys/band1_Tsys.txt
SKA1 = dict(SKA)
SKA2 = dict(SKA)
SKA1["Tsys"] = 30*1e3 #very rough, underestimating at low nu - but for this  analysis okay
SKA1["numin"] = 351 #[MHz]
SKA1["numax"] = 1059

SKA2["Tsys"] = 20*1e3 #overestimating slightly, more like 19 or 18
SKA2["numin"] = 966 #[MHz]
SKA2["numax"] = 1758

hirax = {
    "mode":             'interferometer',
    'Ndish':            1024,                 # No. of dishes
    'Nbeam':            1,                # No. of beams (for multi-pixel detectors)
    'Ddish':            6.,               # Single dish diameter [m]
    "t_int":            17520,               #4 years at 50% obs fraction in [h]
    'Tsys':            50.*(1e3),         # System temp. [mK]
    #'nu_crit':          1000.,            #critical frequency, UNCLEAR
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,             # Max. freq. of survey
    'dnu':              0.4,               # Bandwidth of single channel [MHz]
    'S_area':            1.45*np.pi,          # fsky = 15.000 sq. deg. * (pi/180)**2 Total survey area [radians^2]
    'Dmax':             307.,               # Max. interferom. baseline [m]
    'Dmin':             7.,                 # Min. interferom. baseline [m]
}
cb_hirax=dict(hirax) #same but with actual baseline file.
cb_hirax.update({'n(x)':             "/Users/amadeus/Documents/PhD/work/bao21cm/hirax/hirax_Ndish1024_baseline1.dat"})


hirax512 = dict(hirax)
hirax512.update({ "n(x)": "/Users/amadeus/Documents/PhD/work/bao21cm/hirax/HIRAX_Ndish529_Ndish529_baseline1.dat",
                    "Ndish": 529,
                    } )


LSST = {
    "dNdz":         "/Users/amadeus/Documents/PhD/work/multitracer/LSST/nz_LSST_gold_sqdeg.dat",
    "rmax":27, #see bottom of page 71 in the LSST science book 0912.0201
    "zmax":3.9,
    'sg_file_z': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sg_LSST_z.txt',
    'sgng_file_z': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sgng_LSST_z.txt',
    'sg_file_magmax': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sg_LSST_mag.txt',
    'sgng_file_magmax': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sgng_LSST_mag.txt',
    'sg_file': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sg_LSST.txt',
    'sgng_file': '/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sgng_LSST.txt', #for numerical ease, (5sg-2)*ng
    'zbgmax_file_magmax':    '/Users/amadeus/Documents/PhD/work/MagBias/zbgmax_data/zbgmax_LSST_mag.txt',
    'zbgmax_file':           '/Users/amadeus/Documents/PhD/work/MagBias/zbgmax_data/zbgmax_LSST.txt',
    }
LSST_nosgfit = dict(LSST)# the same but without the sg_file
del LSST_nosgfit["sg_file"]
del LSST_nosgfit["sg_file_z"]
del LSST_nosgfit["sg_file_magmax"]
