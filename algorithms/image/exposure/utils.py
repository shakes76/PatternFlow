import torch
from typing import Optional, Tuple

_integer_types = (torch.uint8, torch.int8, torch.int16,
                  torch.int32, torch.int64)
_integer_ranges = {t: (torch.iinfo(t).min, torch.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {
    torch.uint8: (0, 1),
    torch.float: (-1, 1),
    torch.float16: (-1, 1),
    torch.float32: (-1, 1),
    torch.float64: (-1, 1),
}

dtype_range.update(_integer_ranges)


def interp(x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """perform one-dimentional interpolation

    Return the one-dimentional piece-wise interpolation to a function with
    given discrete data points (xp, yp), evaluated at image

    Parameters
    ----------
    x : torch.Tensor
        the x coordinate at which to evaluate the interpolated values
    xp : torch.Tensor
        The x-coordinates of the data points.
    yp : torch.Tensor
        The y-coordinates of the data points, same length as `xp`.

    Returns
    -------
    torch.Tensor
        The interpolated values, same shape as `x`.

    Notes
    -----
    Does not check that the x-coordinate sequence `xp` is increasing.
    If `xp` is not increasing, the results are nonsense.
    """
    x = x.double()
    xp = xp.double()
    yp = yp.double()

    # these two parameters are used to create linear functions
    slope, b = _calc_coefficients(xp, yp)

    # clamp x in the range of min(xp),max(xp)
    x = torch.clamp(x, min=torch.min(
        xp).item(), max=torch.max(xp).item())

    # find the inverse of linear functions
    inverse = torch.where(slope == 0.0, torch.zeros_like(b), (-b/slope))

    # in order to find where the input value x belongs to which interval
    # I construct a tensor to find the interval
    # for example
    # xp = [1, 2, 3, 4, 5]
    # x = 1.3
    # this function is trying to identify x is between xp[0],xp[1]
    x_value = x.repeat(xp.size()[0]-1, 1).t()
    right_boundary = xp[1:].repeat(x.size()[0], 1)
    right_boundary[:, -1] += 1e-7  # in case x == xp[-1]

    left_boundary = xp[:-1].repeat(x.size()[0], 1)

    less_than = x_value < right_boundary
    greater_than_or_equal = x_value >= left_boundary

    def lerp(xx: torch.Tensor) -> torch.Tensor:
        """
        return a 2D tensor that only contain the interpolated results
        for example
        x: (4,) tensor
        xp: (5, ) tensor
        [
            [result, 0.0, 0.0, 0.0],
            [0.0, result, 0.0, 0.0],
            [0.0, 0.0, 0.0, result],
            [0.0, 0.0, result, 0.0],
        ]
        """
        slope = (yp[1:]-yp[:-1])/(xp[1:]-xp[:-1])
        b = yp[:-1] - slope*xp[:-1]
        result = slope * xx + b
        # handle slope zero case
        result = torch.where(slope == xx, torch.zeros_like(xx), result)
        return result

    result = lerp(torch.where(
        less_than == greater_than_or_equal, x_value, inverse))
    return torch.sum(result, 1)


def _calc_coefficients(xp: torch.Tensor,
                       yp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """calculate the coefficient of linear function or the slope and intercept"""
    slope = (yp[1:]-yp[:-1])/(xp[1:]-xp[:-1])
    b = yp[:-1] - slope*xp[:-1]
    return slope, b


def dtype_limits(image: torch.Tensor,
                 clip_negative: bool = False) -> Tuple[int, int]:
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    clip_negative : Optional[bool], default False
        If True, clip the negative range (i.e. return 0 for min intensity)
    even if the image dtype allows negative values.

    Returns
    -------
    Tuple[int, int]
        lower and upper intensity limits
    """
    imin, imax = dtype_range[image.dtype]
    if clip_negative:
        imin = 0
    return (imin, imax)
