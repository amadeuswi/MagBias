import numpy as np
from magmod import nofz, sg, bgal_new, C_l_DM_CAMB, C_l_HIHI_CAMB, Cl_HIxmag_CAMB


class GalaxyHIMagnification:

    NINT = 5  # coarse because we have fine bins

    def __init__(self, galaxy_hi_experiment, unityweight = False):
        self.experiment = galaxy_hi_experiment
        self.unityweight = unityweight
        self.set_alpha()

    def set_alpha(self):
        self.alpha_minus_one = self._get_alpha_minus_one_tab()

    def _get_n_gal_in_mag_bins(self, z_bg_lower, z_bg_upper):
        z_integrate = np.linspace(
            z_bg_lower, z_bg_upper, self.NINT)
        mtab_edges = self.experiment.magnitude_bin_edges
        dNdztab = np.array([nofz(z_integrate, mmm) for mmm in mtab_edges])
        dNdzdOm = np.trapz(dNdztab, z_integrate, axis=1)

        # now subtract lower from upper
        # already in rad!!!! and multiplying with all sky
        Nz = (dNdzdOm[1:] - dNdzdOm[:-1]) * 4 * np.pi
        Nz = np.atleast_1d(Nz)
        if (Nz < 0).any():
            raise ValueError(
                "cannot have negative number of galaxies, most likely we confused magnitudes here")

        # less than one galaxy is the same as no galaxy...
        Nz[Nz < self.experiment.config_class.N_g_threshold] = 0.

        return Nz

    def _get_alpha_minus_one_tab(self):  # 2(alpha-1) = 5sg-2
        exp = self.experiment.galaxy_instrument
        z = self.experiment.galaxy_redshift_tab + \
            self.experiment.galaxy_redshift_step/2
        maxmag = self.experiment.magnitude_bin_edges[1:]
        res = np.array([(5*sg(z, exp, mmm) - 2)/2 for mmm in maxmag])
        # res[self.n_gal_in_mag_bins < self.experiment.config_class.N_g_threshold,:] = 1.  # exact value doesn't matter, it's multiplied with 0!
        return res

    def get_alpha_minus_one_in_fg_bin(self, idx):
        return self.alpha_minus_one[:, idx]

    def get_W_weight(self, foreground_bin_idx):
        """
        the weight proposed by menard & bartelmann.
        """
        res = self.alpha_minus_one[:, foreground_bin_idx] + \
            np.zeros((self.experiment.n_ell, 1))
        if self.unityweight:
            print("Warning! weight is set to 1")
            return np.ones(res.shape)

        return res

    def _get_S2N_weighted(self, CHImu, CSbg, CDM, biasg, Weight, Ngal, alphaminus1):
        #     num = (2*ltab + 1) * deltaell * fsky
        Cfg = self.get_C_HIHI()
        CSfg = self.experiment.radio_noise
        ltab = self.experiment.ell_tab
        deltaell = self.experiment.delta_ell
        fsky = self.experiment.fsky
        num = (2*ltab + 1) * \
            deltaell * fsky / 2
        denom = 1 + (Cfg + CSfg) * (
            self.average(biasg, Weight, Ngal)**2 * CDM +
            self.average(Weight, Weight, Ngal) * CSbg) / (self.average(CHImu, Weight, Ngal)**2)
        frac = num/denom
        if (frac < 0).any():
            raise ValueError('Negative S2N?!')
    #         frac = np.abs(frac)
        res = np.sum(frac, axis=0)
        return np.sqrt(res)

    def get_galaxy_bias(self, z):
        bias_g_tab = np.array(
            [bgal_new(z, mmm) for mmm in self.experiment.magnitude_bin_edges[1:]])
        # bias_g_tab[self.n_gal_in_mag_bins < self.experiment.config_class.N_g_threshold] = 1.  # exact value doesn't matter, it's multiplied with 0!
        return bias_g_tab

    def get_C_DM(self, z_lower, z_upper):
        return C_l_DM_CAMB(
            self.experiment.ell_tab, z_lower,
            galsurv=self.experiment.galaxy_instrument,
            ZMAX=z_upper)

    def get_C_HIHI(self):
        return C_l_HIHI_CAMB(
            self.experiment.ell_tab,
            self.experiment.lower_redshift,
            self.experiment.upper_redshift
        )

    def get_C_HIxmag(self, z_lower, z_upper):
        zfmean = (self.experiment.lower_redshift +
                  self.experiment.upper_redshift)/2
        delta_zf = (self.experiment.upper_redshift -
                    self.experiment.lower_redshift)/2
        return np.array([
            Cl_HIxmag_CAMB(
                self.experiment.ell_tab,
                zfmean,
                delta_zf,
                z_lower,
                MAXMAG=mmm,
                ZMAX=z_upper,
                NINT_gkernel=self.NINT
            )
            for mmm in self.experiment.magnitude_bin_edges[1:]]).T  # transpose to match shape

    def get_S2N_weighted_in_bg_bin(self, iz):
        zlow = self.experiment.galaxy_redshift_tab[iz]
        zhigh = self.experiment.galaxy_redshift_tab[iz+1]
        zbmean = (zlow + zhigh)/2

        N_g_tab = self._get_n_gal_in_mag_bins(zlow, zhigh)
        # less than one galaxy is the same as no galaxy...
        N_g_tab[N_g_tab < self.experiment.config_class.N_g_threshold] = 0.

        alphaminusone_tab = self.get_alpha_minus_one_in_fg_bin(iz)
        # exact value doesn't matter, it's multiplied with 0!
        alphaminusone_tab[N_g_tab <
                            self.experiment.config_class.N_g_threshold] = 1.

        bias_g_tab = self.get_galaxy_bias(zbmean)
        # exact value doesn't matter, it's multiplied with 0!
        bias_g_tab[N_g_tab < self.experiment.config_class.N_g_threshold] = 1.

        C_DM_tab = self.get_C_DM(zlow, zhigh)
        Cshot = self.experiment.get_shot_noise(zlow, zhigh)
        C_HIxmag_tab = self.get_C_HIxmag(zlow, zhigh)

        Weight = self.get_W_weight(iz)

        return self._get_S2N_weighted(
            C_HIxmag_tab,
            Cshot,
            C_DM_tab,
            bias_g_tab,
            Weight,
            N_g_tab,
            alphaminusone_tab
        )

    def get_S2N_weighted(self):
        S2Ntab = np.array([
            self.get_S2N_weighted_in_bg_bin(iz) for iz in range(len(self.experiment.galaxy_redshift_tab)-1)
            ])
        # for iz in :  # all but the last entry
        #     S2Ntmp = 
        #     S2Ntab.append(S2Ntmp)
        #     if np.isnan(S2Ntmp):
        #         break #no more S2N to get
        #     if iz ==0:
        #         print(S2Ntmp, "S2Ntmp")
        # S2Ntab = np.array(S2Ntab)
        #now squared summation over the background z bins:
        S2Nsqsum = np.sqrt(np.nansum(S2Ntab**2))
        return S2Nsqsum


    @staticmethod
    def average(A, B, N, VECTORINPUT=False):
        """A,B are the arrays to be averaged, same length as N which is the bg number density of galaxies"""
        if VECTORINPUT:
            sumaxis = 0
        else:
            sumaxis = 1
        # A and B are matrices, ell x mag
        return np.sum(A*B*N, axis=sumaxis)/np.sum(N)


if __name__ == '__main__':
    from magbias_experiments import cb_hirax as hirax, SKA, LSST
    from modules.galaxy_hi_experiment import GalaxyHIExperiment

    redshift = (0.7755075, 1.2755075)
    hirax1 = GalaxyHIExperiment(redshift, hirax, LSST)
    mag1 = GalaxyHIMagnification(hirax1)

    print("delta_ell = {}, fsky = {}".format(
        mag1.experiment.delta_ell, mag1.experiment.fsky))
    print("z from {} to {}".format(
        mag1.experiment.lower_redshift, mag1.experiment.upper_redshift))

    print("S2N in this bin: {}".format(mag1.get_S2N_weighted()))
