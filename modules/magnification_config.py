class MagnificationConfig:
    def __init__(self):
        self.redshift_separation = 0.1  # buffer between fg and bg to avoid correlation
        self.rough_ell_resolution = True  # for speedup
        self.max_magnitude = 27
        self.min_magnitude = 19
        self.ell_max = 2200
        self.delta_mag = 0.2

        self.delta_z_bg = 0.1  # subsplitting of LSST bg bin, later added up

        self.N_g_threshold = 1. # threshold of negligibility
