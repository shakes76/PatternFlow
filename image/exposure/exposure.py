import torch
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from utils import dtype_limits


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
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([ 0.25,  0.75]))
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

    sr_dict = {'image': (0, 1), 'dtype': dtype_limits(
        image, clip_negative=False)}
    try:
        hist_range: Tuple[int, int] = sr_dict[str(source_range)]
    except KeyError as e:
        raise ValueError("Wrong value for the `source_range` argument")

    hist = torch.histc(
        image, nbins, min=hist_range[0], max=hist_range[1])

    bin_centers = _calc_bin_centers(hist_range[0], hist_range[1], nbins)

    if normalize:
        hist = hist / torch.sum(hist)

    return (hist, bin_centers)


def _calc_bin_centers(start: Union[int, float] = 0,
                      end: Union[int, float] = 1, nbins: Optional[int] = 2) -> torch.Tensor:
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

    length = abs(start) + abs(end)
    return torch.arange(start, end, step=float(length/nbins)) + float(length/(2*nbins))
