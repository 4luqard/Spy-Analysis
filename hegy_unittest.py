import unittest
import gc

import warnings
def ignore_warn(*args, **kwargs):
    """Suppress all warnings."""
    pass
warnings.warn = ignore_warn

exec(open('hegy.py', 'r').read())
from hegy import hegy_test
gc.collect()

class TestHegy(unittest.TestCase):
    """Unit tests for the HEGY seasonal unit root test."""

    def setUp(self):
        """Set up test data for the tests."""
        self.data = [1, 5, 8, 6, 5, 9, 12, 10, 9, 13, 16, 14, 13, 17, 20, 18, 17, 21, 24, 22, 21, 25, 28, 26]
        
        self.expected_pvalue = {
            "t_statistic": [0.1, 0.08],
            "f_statistic": [0.1, 0.1, 0.1]
        }
        
        self.seasonal_period = 4
        
        self.trend = True
        self.constant = True
        
        self.maxlag = int(round(12 * (len(self.data) / 100) ** (1 / 4)))

    def test_seasonal_unit_root(self):
        # Test the HEGY test on a series with a seasonal unit root
        stat, pvalue = hegy_test(self.data, self.seasonal_period, self.trend, self.constant, self.maxlag)
        for key in self.expected_pvalue:
            for i in range(len(self.expected_pvalue[key])):
                self.assertAlmostEqual(pvalue[key][i], self.expected_pvalue[key][i], places=2)

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()


if __name__ == '__main__':
    unittest.main()