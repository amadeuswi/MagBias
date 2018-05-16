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



SKA = { "Aeff" : 6e5, #[m2]
      "Tsys" : 30, #[K]
      "t_int" : ttot * FOV_SKA / A_SKA,
       "S_area": A_SKA,
       "Name" : "SKA"
      }

CLAR = {
    "Aeff" : 5e4, #[m2]
    "Tsys" : 30, #[K]
    "t_int": ttot * FOV_CLAR / A_CLAR,
    "S_area": A_CLAR,
    "Name" : "CLAR"
}

#
# hirax = {
#     'mode':             'interferom',            # Interferometer or single dish
#     'Ndish':            1024,                 # No. of dishes
#     'Nbeam':            1,                # No. of beams (for multi-pixel detectors)
#     'Ddish':            6.,               # Single dish diameter [m]
#     'Tinst':            50.*(1e3),         # System temp. [mK]
#     #'nu_crit':          1000.,            #critical frequency, UNCLEAR
#     'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
#     'survey_numax':     800.,             # Max. freq. of survey
#     'dnu':              0.4,               # Bandwidth of single channel [MHz]
#     'pointings':        1,                  # number of pointings in drift scan
#     'obs_per_day':      12.,                 #observation hours per day
#     # 'Sarea':            2*np.pi, #half sky    # Total survey area [radians^2]
#     'Dmax':             307.,               # Max. interferom. baseline [m] UNCLEAR
#     'Dmin':             7.,                 # Min. interferom. baseline [m] UNCLEAR
#     'wiggleroom':       1.,                  #wiggle room between dishes
#     'k_nl0':            0.2             # inv Mpc, nonlinear k cutoff
# }
# hirax.update(SURVEY)
# cb_hirax=dict(hirax)
#
# cb_hirax.update({'n(x)':             "/Users/amadeus/Documents/PhD/work/bao21cm/hirax/hirax_Ndish1024_baseline1.dat"})
