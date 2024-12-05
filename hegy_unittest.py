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
        # self.data = [1, 5, 8, 6, 5, 9, 12, 10, 9, 13, 16, 14, 13, 17, 20, 18, 17, 21, 24, 22, 21, 25, 28, 26]
        self.data = [0.9937, 5.0542, 8.0458, 5.9270, 4.9468, 8.9324, 12.0031, 9.9135, 8.9260, 13.0300, 15.9234, 
                     14.0099, 13.0372, 17.0107, 20.0005, 17.9404, 16.9325, 21.0586, 23.9533, 21.9615, 21.0991, 
                     24.9571, 27.9310, 26.0932, 0.9474, 4.9397, 8.0719, 6.0287, 5.0810, 8.9176, 11.9119, 9.9862, 
                     9.0629, 12.9571, 16.0580, 13.9973, 12.9369, 16.9768, 20.0271, 17.9462, 17.0055, 20.9671, 
                     23.9286, 21.9632, 21.0678, 25.0600, 28.0167, 25.9351]
        
        self.expected_pvalue = {
            "p_values_s": [0.101, 0.102],
            "p_values_t": [0.1, 0.91, 0.5, 0.93],
            "p_value_f": 0.95,
        }
        
        self.seasonal_period = 4
        
        self.trend = True
        self.constant = True

    def test_seasonal_unit_root(self):
        # Test the HEGY test on a series with a seasonal unit root
        stat, pvalues = hegy_test(self.data, self.seasonal_period, self.trend, self.constant)
        for key in self.expected_pvalue:
            if key == "p_value_f":
                self.assertAlmostEqual(pvalues[key], self.expected_pvalue[key], places=0)
            else:
                for i in range(len(self.expected_pvalue[key])):
                    self.assertAlmostEqual(pvalues[key][i], self.expected_pvalue[key][i], places=0)

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()


if __name__ == '__main__':
    unittest.main()