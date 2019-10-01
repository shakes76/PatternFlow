from exposure import _calc_bin_centers  # type: ignore
from utils import is_type_integer_family  # type: ignore
# from exposure import _calc_histogram
import unittest
import torch
from skimage import data  # type: ignore


class TestExposure(unittest.TestCase):
    # def test_calc_histogram(self):
    #     image = torch.tensor(data.camera()).float()
    #     expected_vals = [
    #         ValueError,
    #         torch.tensor([262144]),
    #         torch.tensor([107432, 154712]),
    #         torch.tensor([68351, 143206,  50587]),
    #         torch.tensor([63566,  43866, 150517,   4195])
    #     ]

    #     err_msgs = [f'_calc_histogram(image, nbins={i}) should return {expected_val}'
    #                 for i, expected_val in zip(range(5), expected_vals)]

    #     for i in range(5):
    #         with self.subTest(i=i):
    #             if i == 0:
    #                 with self.assertRaises(ValueError, msg=err_msgs[i]):
    #                     _ = _calc_histogram(image, nbins=i)
    #             else:
    #                 hist = _calc_histogram(image, nbins=i)
    #                 self.assertTrue(torch.all(torch.eq(hist, expected_vals[i])),
    #                                 err_msgs[i])

    def test_is_type_integer_family(self):
        int_dtypes = [
            torch.tensor(5, dtype=torch.int8),
            torch.tensor(5, dtype=torch.int16),
            torch.tensor(5, dtype=torch.int32),
            torch.tensor(5, dtype=torch.int64),
        ]

        float_dtypes = [
            torch.tensor(5., dtype=torch.float16),
            torch.tensor(5., dtype=torch.float32),
            torch.tensor(5., dtype=torch.float64),
        ]
        int_err_msgs = [
            f'is_type_integer_family({dtype}) should return true' for dtype in int_dtypes]
        float_err_msgs = [
            f'is_type_integer_family({dtype}) should return false' for dtype in float_dtypes]

        for dtype in int_dtypes:
            with self.subTest(dtype=dtype):
                self.assertTrue(is_type_integer_family(
                    dtype.dtype), msg=int_err_msgs)

        for dtype in float_dtypes:
            with self.subTest(dtype=dtype):
                self.assertFalse(is_type_integer_family(
                    dtype.dtype), msg=float_err_msgs)

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
