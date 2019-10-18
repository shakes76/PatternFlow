"""
COMP3710 Assignment 3

Author: James Copperthwaite

Contrast Limited Adaptive Histogram Equalization (CLAHE).

"""


import tensorflow as tf

# Main Function here
def equalize_adapthist(image, kernel_size=None,
        clip_limit=0.01, nbins=256):
    """
    test
    """

    if kernel_size is None:
        kernel_size = (image.shape[0] // 8, image.shape[1] // 8)
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        ValueError('Incorrect value of `kernel_size`: {}'.format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]
    
    sess = tf.InteractiveSession()

    img = tf.constant(image.astpye(np.uint))
    out = tf.Variabl(img)

    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)
    return rescale_intensity(image)



def _clahe(image, kernel_size, clip_limit, nbins=128):
    """    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    nr = int(np.ceil(image.shape[0] / kernel_size[0])) # numrows
    nc = int(np.ceil(image.shape[1] / kernel_size[1])) # numcolums
    
    # step sizes

def rescale_intensity(image, in_range="image", out_range="dtype"):
    return None


def intensity_range(image, range_values='image', clip_negative=False):
    return None

