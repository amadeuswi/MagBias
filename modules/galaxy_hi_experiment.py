import numpy as np
from numpy import pi
from typing import Tuple

from modules.magnification_config import MagnificationConfig
from magmod import Cl_interferom_noise, noise_cls_single_dish, ztonu21, shotnoise

class GalaxyHIExperiment:

    NINT = 5
    def __init__(self, redshift: Tuple, radio_instrument: dict, galaxy_instrument: dict, magnification_config = MagnificationConfig()):

        self.config_class = magnification_config
        self.lower_redshift = redshift[0]
        self.upper_redshift = redshift[1]
        self.mean_frequency = ztonu21((self.lower_redshift+self.upper_redshift)/2)
        self.radio_instrument = radio_instrument
        self.galaxy_instrument = galaxy_instrument
        self.fsky = self.radio_instrument["S_area"] / (4*np.pi)

        self.lower_galaxy_redshift = self.upper_redshift + self.config_class.redshift_separation
        self.upper_galaxy_redshift = self.galaxy_instrument['zmax']
        self.galaxy_redshift_tab, self.galaxy_redshift_step = self._get_redshift_tab_and_step()

        self.lmin = self._get_ell_min()
        self.lmax = self.config_class.ell_max
        self.delta_ell = self._get_delta_ell()
        self.ell_tab = self._get_ell_tab()
        self.n_ell = len(self.ell_tab)

        self._set_radio_noise()

        if self.config_class.rough_ell_resolution:
            print('we are using rough ell res for speedup')

        self.magnitude_bin_edges = self._get_mag_bin_edges()

    def _get_redshift_tab_and_step(self):
        return np.linspace(
            self.lower_galaxy_redshift,
            self.upper_galaxy_redshift,
            np.int((self.upper_galaxy_redshift-self.lower_galaxy_redshift)/self.config_class.delta_z_bg+1),
            retstep = True
            )

    def _get_ell_min(self):
        if self.radio_instrument['mode'] == 'single_dish':
            res = np.amax([10,np.int(np.around(2*pi/np.sqrt(self.radio_instrument['S_area'])))])
        elif self.radio_instrument['mode'] == 'interferometer':
            res = np.amax([10, np.int(np.around(2*pi/np.sqrt(self._fov())))])
        else:
            raise ValueError('Wrong instrument mode.')
        return res

    def _get_delta_ell(self):
        res = self._get_ell_min()
        if self.config_class.rough_ell_resolution:
            res *= 10
        return res

    def _get_ell_tab(self):
        return np.arange(self.lmin, self.lmax + self.delta_ell, self.delta_ell)

    def _fov(self):
        if self.radio_instrument['mode'] != 'interferometer':
            raise ValueError('fov calculation only implemented for interferometer')
        fov1 = 15 * (pi/180)**2 #this is from table one of the HIRAX white paper
        fov2 = 56 * (pi/180)**2
        mean_fov = (fov1+fov2)/2
        return mean_fov

    def _get_mag_bin_edges(self):
        res = np.arange(self.config_class.min_magnitude, self.config_class.max_magnitude + self.config_class.delta_mag, self.config_class.delta_mag)
        if res[-1] < self.config_class.max_magnitude - 1e-3:
            print(self.config_class.max_magnitude, res[-1])
            raise ValueError("mag bins must extend to max mag.")
        self.n_mag = len(res)
        return res

    def _set_radio_noise(self):
            if self.radio_instrument["mode"] == "single_dish":
                print("Using single dish mode.")
                self.radio_noise = self._single_dish_noise_wrapper()
            elif self.radio_instrument["mode"] == "interferometer":
                print("Using interferometer mode.")
                self.radio_noise = self._interferometer_noise_wrapper()

    def _single_dish_noise_wrapper(self):
        return noise_cls_single_dish(self.ell_tab, self.mean_frequency, self.radio_instrument, 256) * np.ones( self.n_ell )

    def _interferometer_noise_wrapper(self):
        return Cl_interferom_noise(self.ell_tab, self.lower_redshift, self.upper_redshift, self.radio_instrument)


    def get_shot_noise(self, z_lower, z_upper):
        return shotnoise(z_lower, self.galaxy_instrument, self.config_class.max_magnitude, ZMAX = z_upper, NINT = self.NINT)