import torch
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from utils import dtype_limits  # type: ignore
import numpy as np  # type: ignore
from skimage import exposure, data, img_as_float  # type: ignore


def _update_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.int8:
        return torch.int16
    elif dtype == torch.int16:
        return torch.int32
    else:
        return torch.int64


def _offset_array(image: torch.Tensor,
                  lower_boundary: int,
                  higher_boundary: int) -> torch.Tensor:
    """Offset the array to get the lowest value at 0 if negative."""
    if lower_boundary < 0:
        dyn_range = higher_boundary - lower_boundary
        # check if the dyn_range is overflow, if so, update dtype
        if dyn_range > torch.iinfo(image.dtype).max:  # type: ignore
            image = image.type(_update_dtype(image.dtype))
        image = torch.sub(image, lower_boundary)
    return image


def _bin_count_histogram(image: torch.Tensor,
                         source_range: Optional[str] = 'image') -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient histogram calculation for an image of integers.

    This function is significantly more efficient than np.histogram but
    works only on images of integers. It is based on np.bincount.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    source_range : Optional[str], default 'image'
        image: determines the range from the input image
        dtype: determines the range from the expected range of the images
    of that data type.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        hist: The values of the histogram.
        bin_centers: The values at the center of the bins.

    """
    if source_range not in ['image', 'dtype']:
        raise ValueError(
            'Incorrect value for `source_range` argument: {}'.format(source_range))

    if source_range == 'image':
        max_v = int(torch.max(image))
        min_v = int(torch.min(image))
    elif source_range == 'dtype':
        min_v, max_v = dtype_limits(image, clip_negative=False)

    image = _offset_array(image.flatten(), min_v, max_v)
    hist = torch.bincount(image, minlength=max_v-min_v+1)
    bin_centers = torch.arange(min_v, max_v + 1)

    if source_range == 'image':
        idx = max(min_v, 0)
        hist = hist[idx:]
    return hist, bin_centers


def histogram(image: torch.Tensor,
              nbins: Optional[int] = 256,
              source_range: Optional[str] = 'image',
              normalize: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    nbins : Optional[int], default 256
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : Optional[str], default 'image'
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : Optional[bool], default False
        If True, normalize the histogram by the sum of its values.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        hist: The values of the histogram
        bin_centers: The values at the center of the bins.

    Notes
    -----
    cumulative_distribution

    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    >>> image = torch.tensor(img_as_float(data.camera()))
    >>> exposure.histogram(image, nbins=2)
    (tensor([107432, 154712]), tensor([ 0.2500,  0.7500]))
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("input type must be pytorch tensor")

    if not isinstance(nbins, int):
        raise ValueError("given bin must be a integer number")

    shape = image.size()

    if len(shape) == 3 and shape[0] < 4:
        warnings.warn("""This might be a color image. The histogram will be
             computed on the flattened image. You can instead
             apply this function to each color channel.""")

    image = image.flatten()
    min_v = int(torch.min(image))
    max_v = int(torch.max(image))

    if not torch.is_floating_point(image):
        hist, bin_centers = _bin_count_histogram(image, source_range)
    else:
        if source_range == 'image':
            hist = torch.histc(
                image, nbins, min=min_v, max=max_v)
            bin_centers = _calc_bin_centers(min_v, max_v, nbins)
        elif source_range == 'dtype':
            min_v, max_v = dtype_limits(image, clip_negative=False)
            # since the argument of torch.histc min is inclusive, to make is
            # exclusive we have to add a tiny value to it
            hist = torch.histc(
                image, nbins, min=min_v, max=max_v)
            bin_centers = _calc_bin_centers(min_v, max_v, nbins)
        else:
            raise ValueError("Wrong value for the `source_range` argument")

    if normalize:
        hist = torch.div(hist.float(), torch.sum(hist))
        return (hist, bin_centers)

    return (hist.long(), bin_centers)


def _calc_bin_centers(start: Union[int, float] = 0,
                      end: Union[int, float] = 1,
                      nbins: Optional[int] = 2) -> torch.Tensor:
    """calculate the center of bins

    Parameters
    ----------
    start : Union[int, float], default 0
        the starting number of the range
    end : Union[int, float], default 1
        the ending point of the range
    nbins : Optional[int], default 200
        the number of bins that needed

    Returns
    -------
    torch.Tensor
        bin_centers: calculated center of bins
    """
    if not isinstance(nbins, int):
        raise ValueError('number of bins must be integer')

    if start >= end:
        raise ValueError('start cannot greater or equal to end value')

    if nbins < 1:
        raise ValueError('the number of bins cannot less than 1')

    length = end - start
    shift = float(length/nbins)/2.0
    return torch.arange(start, end, step=float(length/nbins)) + shift
