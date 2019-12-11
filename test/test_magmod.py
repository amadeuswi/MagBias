import unittest

from magmod import *

from matplotlib import pyplot as plt


class test_magmod(unittest.TestCase):
    def setUp(self):
        self.ztab = np.linspace(0.01, 2.0, 10)
        self.mtab = np.arange(19, 27 + 1)

    def tearDown(self):
        pass

    def test_sgng_when_lower_mag_interval_is_none_expect_same_as_sgng_interp(self):
        for z in self.ztab:
            for m in self.mtab:
                interpolation_result = sgng_interp(z, m)[0]
                result = sgng(z, [None, m])[0]
                if (interpolation_result is not None
                    and not np.isnan(interpolation_result)
                    and not np.isinf(interpolation_result)
                        and not np.amax((result, interpolation_result)) < 1e-40):
                    self.assertTrue((interpolation_result - result) / interpolation_result < 1e-1)

    def test_dndz_norm_mag_interval_when_lower_mag_interval_is_none_expect_same_as_dndz_norm(self):
        for z in np.linspace(0.5, 2.0, 20):
            for m in self.mtab:
                norm1 = dndz_norm_mag_interval(z, z + 0.1, [None, m])
                norm2 = dndz_norm(z, z + 0.1, m)
                self.assertEqual(norm1, norm2)

    def test_nofz_mag_interval_when_lower_mag_interval_is_none_expect_same_as_nofz(self):
        ztab = np.linspace(0.5, 2.0, 20)
        for m in self.mtab:
            nofz1 = nofz_mag_interval(ztab, [None, m], FIT=False)
            nofz2 = nofz(ztab, m)
            self.assertTrue((nofz1 == nofz2).all())

    # def test_dndz_norm_when_FIT_is_False_expect_same_as_when_FIT_is_true(self):
    #     for z in np.linspace(0.5,2.0,20):
    #         for m in self.mtab:
    #             norm_false = dndz_norm(z, z+0.1, m, FIT = False)
    #             norm_true = dndz_norm(z, z+0.1, m, FIT = True)
    #             print(norm_false / norm_true)

    # def test_visualize_dndz(self):
    #     mag_idx = 0
    #     dndztab1 = dndz_fit(self.ztab, self.mtab[mag_idx])
    #     dndztab2 = nofz(self.ztab, self.mtab[mag_idx])
    #     _, ax = plt.subplots(1,1)
    #     ax.plot(self.ztab, dndztab1)
    #     ax.plot(self.ztab, dndztab2)
    #     plt.show()

    # def test_visualize_dndz_norm(self):
    #     mag_idx = 0
    #     ztab = np.linspace(0.5,2.0,20)
    #     delta_z = 1.
    #     dndznorm1 = np.array([dndz_norm(zz, zz+delta_z, self.mtab[mag_idx], FIT = True) for zz in ztab])
    #     dndznorm2 = np.array([dndz_norm(zz, zz+delta_z, self.mtab[mag_idx], FIT = False) for zz in ztab])
    #     _, ax = plt.subplots(1,1)
    #     ax.plot(ztab, dndznorm1, color = 'black')
    #     ax.plot(ztab, dndznorm2)
    #     plt.show()

    def test_g_with_fit_expect_same_as_without_fit(self):
        zfgtab = np.linspace(0.005, 2.0, 5)
        zbmin = zfgtab[-1] + 0.1
        for m in self.mtab:
            MAG_INTERVAL = [None, m]
            g1 = g(zfgtab, zbmin, MAG_INTERVAL, ZMAX=zbmin + 0.5, NINT=10000, NG_FIT=False, SGNG_FIT=False)
            g2 = g(zfgtab, zbmin, MAG_INTERVAL, ZMAX=zbmin + 0.5, NINT=10000, NG_FIT=True, SGNG_FIT=True)
            if (not np.isnan(g1).any()
                    and not np.isnan(g2).any()
                    and (np.abs(g1) < 1e50).all()
                    and (np.abs(g2) <1e50).all()):
                print((g1 - g2) / g1)


if __name__ == '__main__':
    unittest.main()
