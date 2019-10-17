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
    
    
test_stackcopy()
test_warp_coords_example()
