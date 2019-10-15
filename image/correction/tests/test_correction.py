import unittest
import numpy as np
from PatternFlow.image.correction import correction
from skimage.exposure import exposure


class TestCorrection(unittest.TestCase):
    """
    reference: https://github.com/scikit-image/scikit-image/blob/v0.16.1/skimage/exposure/tests/test_exposure.py
    """

    def test_log_correction_1x1_shape(self):
        img = np.ones([1, 1])
        result = correction.adjust_log(img)
        assert img.shape == result.shape

    def test_log_correction(self):
        """Verifying the output with expected results for logarithmic
        correction with multiplier constant multiplier equal to unity"""
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        expected = exposure.adjust_log(image, 1)
        # expected = np.array([
        #     [0, 5, 11, 16, 22, 27, 33, 38],
        #     [43, 48, 53, 58, 63, 68, 73, 77],
        #     [82, 86, 91, 95, 100, 104, 109, 113],
        #     [117, 121, 125, 129, 133, 137, 141, 145],
        #     [149, 153, 157, 160, 164, 168, 172, 175],
        #     [179, 182, 186, 189, 193, 196, 199, 203],
        #     [206, 209, 213, 216, 219, 222, 225, 228],
        #     [231, 234, 238, 241, 244, 246, 249, 252]], dtype=np.uint8)
        result = correction.adjust_log(image, 1)
        np.testing.assert_array_equal(result, expected)

    def test_adjust_inv_log(self):
        """Verifying the output with expected results for inverse logarithmic
        correction with multiplier constant multiplier equal to unity"""
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        expected = exposure.adjust_log(image, 1, True)
        # expected = np.array([
        #     [  0,   2,   5,   8,  11,  14,  17,  20],
        #     [ 23,  26,  29,  32,  35,  38,  41,  45],
        #     [ 48,  51,  55,  58,  61,  65,  68,  72],
        #     [ 76,  79,  83,  87,  90,  94,  98, 102],
        #     [106, 110, 114, 118, 122, 126, 130, 134],
        #     [138, 143, 147, 151, 156, 160, 165, 170],
        #     [174, 179, 184, 188, 193, 198, 203, 208],
        #     [213, 218, 224, 229, 234, 239, 245, 250]], dtype=np.uint8)
        result = correction.adjust_log(image, 1, True)
        np.testing.assert_array_equal(result, expected)

    def test_adjust_gamma_1x1_shape(self):
        """Check that the shape is maintained"""
        img = np.ones([1,1])
        result = correction.adjust_gamma(img, 1.5)
        assert img.shape == result.shape


    def test_adjust_gamma_one(self):
        """Same image should be returned for gamma equal to one"""
        image = np.random.uniform(0, 255, (8, 8))
        result = correction.adjust_gamma(image, 1)
        np.testing.assert_array_equal(result, image)


    def test_adjust_gamma_zero(self):
        """White image should be returned for gamma equal to zero"""
        image = np.random.uniform(0, 255, (8, 8))
        result = exposure.adjust_gamma(image, 0)
        dtype = image.dtype.type
        assert_array_equal(result, dtype_range[dtype][1])


    def test_adjust_gamma_less_one(self):
        """Verifying the output with expected results for gamma
        correction with gamma equal to half"""
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        expected = np.array([
            [  0,  31,  45,  55,  63,  71,  78,  84],
            [ 90,  95, 100, 105, 110, 115, 119, 123],
            [127, 131, 135, 139, 142, 146, 149, 153],
            [156, 159, 162, 165, 168, 171, 174, 177],
            [180, 183, 186, 188, 191, 194, 196, 199],
            [201, 204, 206, 209, 211, 214, 216, 218],
            [221, 223, 225, 228, 230, 232, 234, 236],
            [238, 241, 243, 245, 247, 249, 251, 253]], dtype=np.uint8)

        result = exposure.adjust_gamma(image, 0.5)
        assert_array_equal(result, expected)


    def test_adjust_gamma_greater_one():
        """Verifying the output with expected results for gamma
        correction with gamma equal to two"""
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        expected = np.array([
            [  0,   0,   0,   0,   1,   1,   2,   3],
            [  4,   5,   6,   7,   9,  10,  12,  14],
            [ 16,  18,  20,  22,  25,  27,  30,  33],
            [ 36,  39,  42,  45,  49,  52,  56,  60],
            [ 64,  68,  72,  76,  81,  85,  90,  95],
            [100, 105, 110, 116, 121, 127, 132, 138],
            [144, 150, 156, 163, 169, 176, 182, 189],
            [196, 203, 211, 218, 225, 233, 241, 249]], dtype=np.uint8)

        result = exposure.adjust_gamma(image, 2)
        assert_array_equal(result, expected)


    def test_adjust_gamma_neggative():
        image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
        with testing.raises(ValueError):
            exposure.adjust_gamma(image, -1)


if __name__ == '__main__':
    unittest.main()
