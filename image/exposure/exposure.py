import torch
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from utils import dtype_limits  # type: ignore
from utils import interp  # type: ignore
from utils import dtype_range
import numpy as np  # type: ignore
from skimage import exposure, data, img_as_float  # type: ignore
import functools


DTYPE_RANGE = dtype_range.copy()


def tensor_image_checker(func):
    """a tensor type checker decorator helper function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if type(args[0]) != torch.Tensor:
            raise TypeError(
                "exposure functions only support data type of pytorch tensor.")
        return func(*args, **kwargs)
    return wrapper


def _update_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.int8:
        return torch.int16
    elif dtype == torch.int16:
        return torch.int32
    else:
        return torch.int64


@tensor_image_checker
def _offset_array(image: torch.Tensor,
                  lower_boundary: int,
                  higher_boundary: int) -> torch.Tensor:
    """Offset the array to get the lowest value at 0 if negative."""
    if lower_boundary < 0:
        dyn_range = higher_boundary - lower_boundary
        # check if the dyn_range is overflow, if so, update dtype
        if dyn_range > torch.iinfo(image.dtype).max:  # type: ignore
            image = image.type(_update_dtype(image.dtype))  # type: ignore
        image = torch.sub(image, lower_boundary)
    return image


@tensor_image_checker
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
    if not isinstance(nbins, int):
        raise ValueError("given bin must be a integer number")

    shape = image.size()

    if len(shape) == 3 and shape[0] < 4:
        warnings.warn("""This might be a color image. The histogram will be
             computed on the flattened image. You can instead
             apply this function to each color channel.""")

    image = image.flatten()
    min_v = float(torch.min(image))
    max_v = float(torch.max(image))

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
    >>> image = img_as_float(data.camera())
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> np.alltrue(cdf[0] == np.cumsum(hi[0])/float(image.size))
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
        Array of same shape as `image`. Only points at which mask == True
        are used for the equalization, which is applied to the whole image.

    Returns
    -------
    torch.Tensor
        Image array after histogram equalization

    Notes
    -----
    This function is adapted from [1]_ with the author's permission.

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


def intensity_range(image: torch.Tensor,
                    range_values: Optional[Union[Tuple[float, float], str]] = 'image',
                    clip_negative: Optional[bool] = False) -> Tuple[float, float]:
    """Return image intensity range (min, max) based on desired value type.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    range_values : Optional[str], default 'image'
        The image intensity range is configured by this parameter.
        The possible values for this parameter are enumerated below.

        'image'
            Return image min/max as the range.
        'dtype'
            Return min/max of the image's dtype as the range.
        dtype-name
            Return intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`. Note: `image` is ignored for this range type.
        2-tuple
            Return `range_values` as min/max intensities. Note that there's no
            reason to use this function if you just want to specify the
            intensity range explicitly. This option is included for functions
            that use `intensity_range` to support all desired range types.

    clip_negative : Optional[bool], default False
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    """
    dtype: Union[torch.dtype, str]
    if range_values == 'dtype':
        dtype = image.dtype
    else:
        dtype = '_'

    if range_values == 'image':
        i_min = torch.min(image).item()
        i_max = torch.max(image).item()
    elif dtype in DTYPE_RANGE:
        if isinstance(dtype, torch.dtype):
            i_min, i_max = DTYPE_RANGE[dtype]
        if clip_negative:
            i_min = 0
    else:
        if isinstance(range_values, tuple):
            i_min, i_max = range_values
        else:
            raise ValueError("range_values must be a tuple")
    return i_min, i_max


def rescale_intensity(image: torch.Tensor,
                      in_range: Union[Tuple[float, float], str] = 'image',
                      out_range: Union[Tuple[float, float], str] = 'dtype') -> torch.Tensor:
    """Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image. See examples below.

    Parameters
    ----------
    image : torch.Tensor
        Input image
    in_range, out_range : Optional[Union[Tuple[float, float], str]], default 'image'
        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.
        'image'
        Use image min/max as the intensity range.
        'dtype'
        Use min/max of the image's dtype as the intensity range.
        dtype-name
        Use intensity range based on desired `dtype`. Must be valid key
        in `DTYPE_RANGE`.
        2-tuple
        Use `range_values` as explicit min/max intensities.

    Returns
    -------
    torch.Tensor
        out: Image tensor after rescaling its intensity. This image is the same dtype
        as the input image.

    See also
    -----
    equalize_hist

    Examples
    --------
    By default, the min/max intensities of the input image are stretched to
    the limits allowed by the image's dtype, since `in_range` defaults to
    'image' and `out_range` defaults to 'dtype':
        >>> image = np.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)
    It's easy to accidentally convert an image dtype from uint8 to float:
    >>> 1.0 * image
    array([  51.,  102.,  153.])
    Use `rescale_intensity` to rescale to the proper range for float dtypes:
    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([ 0. ,  0.5,  1. ])
    To maintain the low contrast of the original, use the `in_range` parameter:
    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([ 0.2,  0.4,  0.6])
    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:
    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([ 0.5,  1. ,  1. ])
    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter:
    >>> image = np.array([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0,  63, 127], dtype=int8)
    """
    dtype = image.dtype

    imin, imax = intensity_range(image, in_range)
    omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))

    image = torch.clamp(image, imin, imax)

    image = (image - imin) / float(imax-imin)
    return torch.tensor(image * (omax-omin) + omin, dtype=dtype)


if __name__ == "__main__":
    args = ['image', 'dtype', (0.3, 0.4)]
    expects = [(0.1, 0.2), (-1, 1), (0.3, 0.4)]
    image = torch.tensor([0.1, 0.2], dtype=torch.double)
    for arg, expect in zip(args, expects):
        out = intensity_range(image, range_values=arg)
        print(out)
