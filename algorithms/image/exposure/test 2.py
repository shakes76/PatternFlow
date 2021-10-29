import unittest
import torch
from skimage import data, img_as_float, exposure, img_as_ubyte
import numpy as np
from utils import dtype_range
from exposure import (  # type: ignore
    _calc_bin_centers,
    histogram,
    equalize_hist,
    cumulative_distribution,
    adjust_gamma,
)


class TestGammaCorrection(unittest.TestCase):
    def test_adjust_gamma_one(self):
        """Same image should be returned for gamma equal to one"""
        image = torch.FloatTensor(8, 8).uniform_(0, 255)
        result = adjust_gamma(image, 1)
        self.assertTrue(torch.equal(result, image))

    def test_adjust_gamma_zero(self):
        """White image should be returned for gamma equal to zero"""
        image = torch.FloatTensor(8, 8).uniform_(0, 255)
        result = adjust_gamma(image, 0)
        dtype = image.dtype
        self.assertTrue(torch.all(result == torch.tensor(
            dtype_range[dtype][1], dtype=result.dtype)))

    def test_adjust_gamma_less_one(self):
        """Verifying the output with expected results for gamma
        correction with gamma equal to half"""
        image = torch.arange(0, 255, 4).reshape((8, 8)).type(torch.uint8)
        expected = torch.tensor([
            [0,  31,  45,  55,  63,  71,  78,  84],
            [90,  95, 100, 105, 110, 115, 119, 123],
            [127, 131, 135, 139, 142, 146, 149, 153],
            [156, 159, 162, 165, 168, 171, 174, 177],
            [180, 183, 186, 188, 191, 194, 196, 199],
            [201, 204, 206, 209, 211, 214, 216, 218],
            [221, 223, 225, 228, 230, 232, 234, 236],
            [238, 241, 243, 245, 247, 249, 251, 253]], dtype=torch.uint8)

        result = adjust_gamma(image, 0.5)
        self.assertTrue(torch.equal(result, expected))

    def test_adjust_gamma_greater_one(self):
        """Verifying the output with expected results for gamma
        correction with gamma equal to two"""
        image = torch.arange(0, 255, 4).reshape((8, 8)).type(torch.uint8)
        expected = torch.tensor([
            [0,   0,   0,   0,   1,   1,   2,   3],
            [4,   5,   6,   7,   9,  10,  12,  14],
            [16,  18,  20,  22,  25,  27,  30,  33],
            [36,  39,  42,  45,  49,  52,  56,  60],
            [64,  68,  72,  76,  81,  85,  90,  95],
            [100, 105, 110, 116, 121, 127, 132, 138],
            [144, 150, 156, 163, 169, 176, 182, 189],
            [196, 203, 211, 218, 225, 233, 241, 249]], dtype=torch.uint8)

        result = adjust_gamma(image, 2)
        self.assertTrue(torch.equal(result, expected))


class TestEqualization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEqualization, self).__init__(*args, **kwargs)
        torch.manual_seed(0)
        self.test_img_int = torch.tensor(data.camera())
        self.test_img = img_as_float(self.test_img_int)
        self.test_img = exposure.rescale_intensity(self.test_img / 5. + 100)

    def test_equalize_uint8_approx(self):
        """Check integer bins used for uint8 images."""
        img_eq0 = equalize_hist(self.test_img_int)
        img_eq1 = equalize_hist(self.test_img_int, nbins=3)
        self.assertTrue(torch.allclose(img_eq0, img_eq1))

    def test_equalize_ubyte(self):
        with self.assertWarns(UserWarning):
            img = torch.tensor(img_as_ubyte(self.test_img))
        img_eq = equalize_hist(img)

        cdf, bin_edges = cumulative_distribution(img_eq)
        self.check_cdf_slope(cdf)

    def test_equalize_float(self):
        img = torch.tensor(img_as_float(self.test_img))
        img_eq = equalize_hist(img)

        cdf, bin_edges = cumulative_distribution(img_eq)
        self.check_cdf_slope(cdf)

    def test_equalize_masked(self):
        img = torch.tensor(img_as_float(self.test_img))
        mask = torch.zeros(self.test_img.shape, dtype=torch.bool)
        mask[50:150, 50:250] = 1
        img_mask_eq = equalize_hist(img, mask=mask)
        img_eq = equalize_hist(img)

        cdf, bin_edges = cumulative_distribution(img_mask_eq)
        self.check_cdf_slope(cdf)

        self.assertFalse(torch.equal(img_eq, img_mask_eq))

    def check_cdf_slope(self, cdf):
        """Slope of cdf which should equal 1 for an equalized histogram."""
        norm_intensity = torch.linspace(0, 1, len(cdf))
        slope, intercept = np.polyfit(norm_intensity.numpy(), cdf.numpy(), 1)
        assert 0.9 < slope < 1.1


class TestHistogram(unittest.TestCase):
    def test_input_tensor(self):
        image = [10, 100]
        with self.assertRaises(TypeError):
            _, _ = histogram(image)

    def test_wrong_source_range(self):
        image = torch.tensor([-1, 100], dtype=torch.int8)
        with self.assertRaises(ValueError):
            _, _ = histogram(image, source_range='foobar')

    def test_negative_overflow(self):
        image = torch.tensor([-1, 100], dtype=torch.int8)
        hist, bin_centers = histogram(image)
        self.assertTrue(torch.equal(bin_centers, torch.arange(-1, 101)))
        self.assertTrue(hist[0] == 1)
        self.assertTrue(hist[-1] == 1)
        self.assertTrue((hist[1:-1] == 0).all())

    def test_negative_image(self):
        image = torch.tensor([-100, -1], dtype=torch.int8)
        hist, bin_centers = histogram(image)
        self.assertTrue(torch.equal(bin_centers, torch.arange(-100, 0)))
        self.assertTrue(hist[0] == 1)
        self.assertTrue(hist[-1] == 1)
        self.assertTrue((hist[1:-1] == 0).all())

    def test_int_range_image(self):
        image = torch.tensor([10, 100], dtype=torch.int8)
        hist, bin_centers = histogram(image)
        self.assertEqual(len(hist), len(bin_centers))
        self.assertEqual(bin_centers[0], 10)
        self.assertEqual(bin_centers[-1], 100)

    def test_peak_uint_range_dtype(self):
        image = torch.tensor([10, 100], dtype=torch.int8)
        hist, bin_centers = histogram(image)
        self.assertEqual(len(hist), len(bin_centers))
        self.assertEqual(bin_centers[0], 10)
        self.assertEqual(bin_centers[-1], 100)

    def test_peak_int_range_dtype(self):
        image = torch.tensor([10, 100], dtype=torch.int8)
        hist, bin_centers = histogram(image, source_range='dtype')
        self.assertTrue(torch.equal(bin_centers, torch.arange(-128, 128)))
        self.assertEqual(hist[128+10], 1)
        self.assertEqual(hist[128+100], 1)
        self.assertEqual(hist[128+101], 0)
        self.assertEqual(hist.size(), torch.Size([256]))

    def test_flat_uint_range_dtype(self):
        image = torch.linspace(0, 255, 256).type(torch.uint8)
        hist, bin_centers = histogram(image, source_range='dtype')
        self.assertTrue(torch.equal(bin_centers, torch.arange(0, 256)))
        self.assertEqual(hist.size(), torch.Size([256]))

    def test_flat_int_range_dtype(self):
        image = torch.linspace(-128, 128, 256).type(torch.int8)
        hist, bin_centers = histogram(image, source_range='dtype')
        self.assertTrue(torch.equal(bin_centers, torch.arange(-128, 128)))
        self.assertEqual(hist.size(), torch.Size([256]))

    def test_peak_float_out_of_range_image(self):
        image = torch.tensor([10, 100], dtype=torch.float)
        hist, bin_centers = histogram(image, nbins=90)
        # offset values by 0.5 for float...
        self.assertTrue(torch.equal(
            bin_centers, torch.arange(10, 100).float().add(0.5)))

    def test_peak_float_out_of_range_dtype(self):
        image = torch.tensor([10, 100], dtype=torch.float)
        hist, bin_centers = histogram(image, nbins=10, source_range='dtype')
        self.assertTrue(torch.allclose(
            torch.min(bin_centers), torch.tensor(-0.9)))
        self.assertTrue(torch.allclose(
            torch.max(bin_centers), torch.tensor(0.9)))
        self.assertEqual(len(bin_centers), 10)

    def test_normalize(self):
        image = torch.tensor([0, 255, 255], dtype=torch.uint8)
        hist, bin_centers = histogram(image, source_range='dtype',
                                      normalize=False)
        expected = torch.zeros(256, dtype=torch.long)
        expected[0] = 1
        expected[-1] = 2
        self.assertTrue(torch.equal(hist, expected))
        hist, bin_centers = histogram(image, source_range='dtype',
                                      normalize=True)
        expected = expected.float().div(3.0)
        self.assertTrue(torch.equal(hist, expected))

    def test_calc_bin_centers(self):
        # the arguments passed to _calc_bin_centers
        # args[0]: start
        # args[1]: end
        # args[2]: bins
        args = [
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (-1, 1, 1),
            (-1, 1, 2),
            (-1, 1, 4),
            (2, 27, 5),
            (-10, 10, 5),
        ]
        # the expected return value
        expected_values = [
            ValueError,
            torch.tensor([0.5]),
            torch.tensor([0.25, 0.75]),
            torch.tensor([0.0]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([-0.75, -0.25, 0.25, 0.75]),
            torch.tensor([4.5, 9.5, 14.5, 19.5, 24.5]),
            torch.tensor([-8.0, -4.0, 0.0, 4.0, 8.0]),
        ]

        # the error message should be returned when not match
        err_msgs = [f'_cal_bin_center{args} should return {str(expected_value)}'
                    for args, expected_value in zip(args, expected_values)]

        # create test cases based on function arguments, the expected result
        # and the error message
        test_cases = dict(zip(args, zip(expected_values, err_msgs)))

        for args, expected_val_and_err_msg in test_cases.items():
            with self.subTest(args=args, expected_val_and_err_msg=expected_val_and_err_msg):
                # test the ValueError case
                if type(expected_val_and_err_msg[0]) != torch.Tensor:
                    with self.assertRaises(expected_val_and_err_msg[0],
                                           msg=expected_val_and_err_msg[1]):
                        _ = _calc_bin_centers(*args)
                # test other cases
                else:
                    calc = _calc_bin_centers(*args)
                    is_equal = torch.all(
                        torch.eq(calc, expected_val_and_err_msg[0]))
                    self.assertTrue(is_equal, expected_val_and_err_msg[1])


if __name__ == "__main__":
    unittest.main()
