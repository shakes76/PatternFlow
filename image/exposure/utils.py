import torch
from typing import Optional, Tuple

dtype_range = {
    torch.float: (-1, 1),
    torch.float16: (-1, 1),
    torch.float32: (-1, 1),
    torch.int8: (0, 1),
    torch.uint8: (0, 1),
    torch.short: (-1, 1),
    torch.int: (-1, 1),
    torch.long: (-1, 1),
}


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
    imin, imax = (-1, 1)
    if clip_negative:
        imin = 0
    return (imax, imin)


def is_type_integer_family(dtype: torch.dtype) -> bool:
    return (dtype == torch.uint8 or
            dtype == torch.int8 or
            dtype == torch.int16 or
            dtype == torch.int32 or
            dtype == torch.int64)
