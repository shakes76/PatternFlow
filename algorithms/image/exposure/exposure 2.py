import torch
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from utils import dtype_limits, interp, dtype_range
import functools


DTYPE_RANGE = dtype_range.copy()

################################################################################
############################  main algorithms  #################################
################################################################################


def tensor_image_checker(func):
    """a tensor type checker decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if type(args[0]) != torch.Tensor:
            raise TypeError(
                "exposure functions only support data type of pytorch tensor.")
        return func(*args, **kwargs)
    return wrapper


@tensor_image_checker
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
        bin_centers: The values at the center of bins.

    Notes
    -----
    cumulative_distribution

    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> import torch
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    >>> image = torch.tensor(img_as_float(data.camera()))
    >>> exposure.histogram(image, nbins=2)
    (tensor([107432, 154712]), tensor([ 0.2500,  0.7500]))
    """
    if not isinstance(nbins, int):
        raise ValueError("Given bin cannot be non integer type")

    shape = image.size()

    if len(shape) == 3 and shape[0] < 4:
        warnings.warn("""This might be a color image. The histogram will be
             computed on the flattened image. You can instead
             apply this function to each color channel.""")

    image = image.flatten()
    min_v = torch.min(image).item()
    max_v = torch.max(image).item()

    # if the input image is normal integer type
    # like gray scale from 0-255, we implement fast histogram calculation
    # by returning bin count for each pixel value
    if not torch.is_floating_point(image):
        hist, bin_centers = _bin_count_histogram(image, source_range)
    else:
        if source_range == 'image':
            hist = torch.histc(
                image, nbins, min=min_v, max=max_v)
            bin_centers = _calc_bin_centers(min_v, max_v, nbins)
        elif source_range == 'dtype':
            min_v, max_v = dtype_limits(image, clip_negative=False)
            hist = torch.histc(
                image, nbins, min=min_v, max=max_v)
            bin_centers = _calc_bin_centers(min_v, max_v, nbins)
        else:
            raise ValueError("Wrong value for the `source_range` argument")

    if normalize:
        hist = torch.div(hist, float(torch.sum(hist).item()))
        return (hist, bin_centers)

    return (hist.long(), bin_centers)


@tensor_image_checker
def cumulative_distribution(image: torch.Tensor,
                            nbins: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cumulative distribution function (cdf) for the given image.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    nbins : int, default 256
        Number of bins for image histogram

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        img_cdf: value of cumulative distribution function.
        bin_centers: centers of bins

    See Also
    -----
    histogram

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function

    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> import torch
    >>> image = torch.tensor(img_as_float(data.camera()))
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> np.alltrue(cdf.numpy()[0] == np.cumsum(hi[0])/float(image.size))
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = torch.cumsum(hist, dim=0)
    img_cdf = torch.div(img_cdf.double(), img_cdf[-1])
    return img_cdf, bin_centers


@tensor_image_checker
def equalize_hist(image: torch.Tensor,
                  nbins: int = 256,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return image after histogram equalization.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    nbins : int, default 256
        Number of bins for image histogram. Note: this argument is
        ignored for integer images, for which each integer is its own
        bin.
    mask : [type], default None
        Pytorch Tensor of same shape as `image`. Only points at which mask == True
        are used for the equalization, which is applied to the whole image.

    Returns
    -------
    torch.Tensor
        Image tensor after histogram equalization

    References
    ----------
    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization
    """
    if mask is not None:
        cdf, bin_centers = cumulative_distribution(image[mask], nbins)
    else:
        cdf, bin_centers = cumulative_distribution(image, nbins)
    out = interp(image.flatten(), bin_centers, cdf)
    return out.reshape(image.shape)


@tensor_image_checker
def adjust_gamma(image: torch.Tensor,
                 gamma: float = 1.0,
                 gain: float = 1.0) -> torch.Tensor:
    """Performs Gamma Correction on the input image.

    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    gamma : float, default 1.0
        Non negative real number. Default value is 1.
    gain : float, default 1.0
        The constant multiplier. Default value is 1.

    Returns
    -------
    torch.Tensor
        Gamma corrected output image.

    See also
    -----
    adjust_log

    Notes
    --------
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gamma_correction

    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> import torch
    >>> image = torch.tensor(img_as_float(data.moon()))
    >>> gamma_corrected = exposure.adjust_gamma(image, 2)
    >>> # Output is darker for gamma > 1
    >>> image.mean() > gamma_corrected.mean()
    True
    """
    dtype: torch.dtype = image.dtype

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    out = ((image / scale) ** gamma) * scale * gain

    return out.type(dtype)

################################################################################
############################  helper functions  ################################
################################################################################


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
        raise ValueError('number of bins is not integer')

    if start >= end:
        raise ValueError('start cannot greater or equal to end value')

    if nbins < 1:
        raise ValueError('the number of bins cannot less than 1')

    width = end - start
    shift = float(width/nbins)/2.0
    return torch.arange(start, end, step=float(width/nbins)) + shift


def _update_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.int8:
        return torch.int16
    elif dtype == torch.int16:
        return torch.int32
    else:
        return torch.int64


def _offset_array(image: torch.Tensor,
                  lower_boundary: float,
                  higher_boundary: float) -> torch.Tensor:
    """Offset the array to get the lowest value at 0 if negative."""
    if lower_boundary < 0:
        dyn_range = higher_boundary - lower_boundary
        # check if the dyn_range is overflow, if so, update dtype
        if dyn_range > torch.iinfo(image.dtype).max:
            image = image.type(_update_dtype(image.dtype))
        image = torch.sub(image, lower_boundary)
    return image


def _bin_count_histogram(image: torch.Tensor,
                         source_range: Optional[str] = 'image') -> Tuple[torch.Tensor,
                                                                         torch.Tensor]:
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
            f'Incorrect value for `source_range` argument: {source_range}')

    if source_range == 'image':
        max_v = torch.max(image).item()
        min_v = torch.min(image).item()
    elif source_range == 'dtype':
        min_v, max_v = dtype_limits(image, clip_negative=False)

    image = _offset_array(image.flatten(), min_v, max_v)
    hist = torch.bincount(image, minlength=int(max_v-min_v + 1))
    bin_centers = torch.arange(min_v, max_v + 1)

    if source_range == 'image':
        idx: int = int(max(min_v, 0))
        hist = hist[idx:]
    return hist, bin_centers
