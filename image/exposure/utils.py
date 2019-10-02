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
}

dtype_range.update(_integer_ranges)


def dtype_limits(image: torch.Tensor,
                 clip_negative: Optional[bool] = False) -> Tuple[int, int]:
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
