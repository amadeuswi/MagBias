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
