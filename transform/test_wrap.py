import warp_coords as tf_wrap
from numpy.testing import assert_almost_equal
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import transform as tf, data
from skimage.transform import SimilarityTransform, warp_coords
from matplotlib import pyplot as plt
import tensorflow as tf

def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    tf_wrap._stackcopy(x, y)
    for i in range(layers):
        assert_almost_equal(x[..., i], y)
    
def shift_up10_left20(xy):
    return xy - np.array([-20, 10])[None, :]

def test_warp_coords_example():
    image = data.astronaut().astype(np.float32)
    tf_coords = tf_wrap.warp_coords(shift_up10_left20, image.shape)
    tf_warped_image = map_coordinates(image, tf_coords)
    
    coords = warp_coords(shift_up10_left20, image.shape)
    warped_image = map_coordinates(image, coords)
    assert_almost_equal(tf_warped_image , warped_image)
    
    _debug_plot(warped_image, tf_warped_image)
    
test_stackcopy()
test_warp_coords_example()

def _debug_plot(original, result, sinogram=None):
    from matplotlib import pyplot as plt
    imkwargs = dict(cmap='gray', interpolation='nearest')
    if sinogram is None:
        plt.figure(figsize=(15, 6))
        sp = 130
    else:
        plt.figure(figsize=(11, 11))
        sp = 221
        plt.subplot(sp + 0)
        plt.imshow(sinogram, aspect='auto', **imkwargs)
    plt.subplot(sp + 1)
    plt.imshow(original, **imkwargs)
    plt.subplot(sp + 2)
    plt.imshow(result, vmin=original.min(), vmax=original.max(), **imkwargs)
    plt.subplot(sp + 3)
    plt.imshow(result - original, **imkwargs)
    plt.colorbar()
    plt.show()