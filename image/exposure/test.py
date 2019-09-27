from exposure import _calc_bin_centers
import unittest
import torch


class TestExposure(unittest.TestCase):
    def test_calc_bin_centers(self):
        # the arguments passed to _calc_bin_centers
        # args[0]: start
        # args[1]: end
        # args[2]: bins
        args = [
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (-1, 1, 2),
            (-1, 1, 1)
        ]
        # the expected return value
        expected_values = [
            ValueError,
            torch.tensor([0.5]),
            torch.tensor([0.25, 0.75]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.0])
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
                    with self.assertRaises(expected_val_and_err_msg[0], msg=expected_val_and_err_msg[1]):
                        _ = _calc_bin_centers(*args)
                # test other cases
                else:
                    calc = _calc_bin_centers(*args)
                    is_equal = torch.all(
                        torch.eq(calc, expected_val_and_err_msg[0]))
                    self.assertTrue(is_equal, expected_val_and_err_msg[1])


if __name__ == "__main__":
    unittest.main()
