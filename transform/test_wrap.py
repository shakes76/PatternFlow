from warp_coords import warp_coords, _stackcopy
from numpy.testing import assert_almost_equal
import numpy as np
import tensorflow as tf
def test_stackcopy():
    sess = tf.InteractiveSession()
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_almost_equal(x[..., i], y)
    
test_stackcopy()