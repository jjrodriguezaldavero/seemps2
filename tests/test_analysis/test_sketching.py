import numpy as np
from scipy.stats import norm
from ..tools import TestCase

from seemps.state import DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval, Mesh, mps_to_mesh_matrix
from seemps.analysis.sketching import tt_rss, BlackBoxLoadMPS

class TestTTRSS(TestCase):
    def test_1d_gaussian(self):
        n = 14
        N = 2**n
        a, b = -1.0, 1.0
        interval = RegularInterval(a, b, N)
        x = interval.to_vector()

        loc, scale = 0.0, 1.0
        f = lambda x: norm.pdf(x, loc=loc, scale=scale) # noqa: E731
        y_vec = f(x)

        num_samples = 100
        samples = np.random.normal(loc=loc, scale=scale, size=num_samples).reshape(-1, 1)
        
        map_matrix = mps_to_mesh_matrix([n])
        physical_dimensions = [2] * n
        black_box = BlackBoxLoadMPS(f, Mesh([interval]), map_matrix, physical_dimensions)

        max_bonds = np.array([10]*n, dtype=int)
        mps = tt_rss(black_box, samples, max_bonds, strategy=DEFAULT_STRATEGY)
        y_mps = mps.to_vector()

        self.assertTrue(np.allclose(y_vec, y_mps, atol=1e-7))