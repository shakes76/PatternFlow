"""
COMP3710 Assignment 3

Author: James Copperthwaite

Contrast Limited Adaptive Histogram Equalization (CLAHE).

"""


import tensorflow as tf
import numpy as np

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
    out = tf.Variable(img)

    tf.global_variables_initializer().run() #init variables

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

def histogram(image, nbins=12, normalize=False):
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")

    image = image.flatten()
    # For integer types, histogramming with bincount is more efficient.

    hist, bin_edges = np.histogram(image, bins=nbins, range=None)
    print(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    print("MAX", np.max(image))
    print("MIN", np.min(image))
    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers

def tfhist(image, nbins=12, normalize=False):
    sess = tf.InteractiveSession()
    
    img = tf.constant(image.astype(np.float32))
    hist = tf.Variable(np.zeros(nbins).astype(np.int64))
    out = tf.Variable(img)

    tf.global_variables_initializer().run() #init variables

    img = tf.reshape(img, [-1]) 

    current_min = tf.dtypes.cast(tf.reduce_min(img), tf.float32)
    current_max = tf.dtypes.cast(tf.reduce_max(img), tf.float32)
    
    bins = tf.linspace(current_min, current_max, nbins+1)
    
    y, idx, count = tf.unique_with_counts(img)
    
    
    for i in range(nbins):
        if i==nbins-1:
            # if last bin cut off at less than or equal to the bin limit
            mask = (img <= bins[i+1]) 
            lim = tf.boolean_mask(img, mask)
            
        else: # cut off bin at less than upper bound 
            mask = (img < bins[i+1]) 
            lim = tf.boolean_mask(img, mask)
        mask = (lim >= bins[i]) # all bins are greater than or equal to lower bound
        lim = tf.boolean_mask(img, mask)

        vals = tf.dtypes.cast(mask, tf.int32)
        count = tf.count_nonzero(vals)

        idx = tf.dtypes.cast(tf.one_hot(i, nbins), tf.int64)  # output: [3 x 3]
        idx = tf.math.scalar_mul(count, idx)
        hist = hist + idx

    mids = (bins[:-1] + bins[1:]) / 2
    
    h = hist.eval()
    bin_centers = mids.eval()
    sess.close()
    return h, bin_centers


    


