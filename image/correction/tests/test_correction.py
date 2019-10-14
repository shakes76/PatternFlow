import unittest
import numpy as np
from PatternFlow.image.correction import correction
from skimage.exposure import exposure


class TestCorrection(unittest.TestCase):

    def test_log_correction_1x1_shape(self):
        img = np.ones([1, 1])
        result = correction.adjust_log(img)
        assert img.shape == result.shape

    def test_log_correction(self):
        """Verifying the output with expected results for logarithmic
        correction with multiplier constant multiplier equal to unity"""
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        expected = np.array([
            [0, 5, 11, 16, 22, 27, 33, 38],
            [43, 48, 53, 58, 63, 68, 73, 77],
            [82, 86, 91, 95, 100, 104, 109, 113],
            [117, 121, 125, 129, 133, 137, 141, 145],
            [149, 153, 157, 160, 164, 168, 172, 175],
            [179, 182, 186, 189, 193, 196, 199, 203],
            [206, 209, 213, 216, 219, 222, 225, 228],
            [231, 234, 238, 241, 244, 246, 249, 252]], dtype=np.uint8)
        expected = exposure.adjust_log(image, 1)
        result = correction.adjust_log(image, 1)
        np.testing.assert_almost_equal(result, expected)




if __name__ == '__main__':
    unittest.main()
