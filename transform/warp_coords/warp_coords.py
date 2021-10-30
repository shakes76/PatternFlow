# In[1]: Import
import tensorflow as tf
# In[2]: wrap_ coords
def warp_coords(coord_map, shape, dtype=tf.float64):
    """Build the source coordinates for the output of a 2-D image warp.

    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : tf.dtype or string
        dtype for return value (sane choices: float32 or float64).

    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.

    Notes
    -----

    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.

    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.


    Examples
    --------
    
    def shift_down10_left20(xy):
        return xy - np.array([-20, 10])[None, :]
    
    def test_warp_coords_example():
        ##
        image = data.astronaut().astype(np.float32)
        tf_coords = tf_wrap.warp_coords(shift_down10_left20, image.shape, dtype=tf.float32)
        tf_warped_image = map_coordinates(image, tf_coords)
        
        coords = warp_coords(shift_down10_left20, image.shape, dtype=np.float32)
        warped_image = map_coordinates(image, coords)
        assert_almost_equal(tf_warped_image , warped_image)
        plot_result(image, tf_warped_image)
        
    
    def plot_result(original, result):
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax = axes.ravel()
        ax[0].imshow(original)
        ax[0].set_title("Original image")
        ax[1].imshow(result)
        ax[1].set_title("Shifted Image by 20 Left and 10 Down")
        plt.tight_layout()
        plt.show()
    """
    sess = tf.Session()
    with sess.as_default():
        
        rows, cols = shape[0], shape[1]
        coords_shape = [len(shape), rows, cols]
        if len(shape) == 3:
            coords_shape.append(shape[2])
            
        # Initialise an empty coords with same shape and datatype
        coords = tf.zeros(coords_shape, dtype=dtype)
        
        # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
        x, y = tf.meshgrid(tf.range(cols), tf.range(rows))
        x = tf.cast(x, dtype)
        y = tf.cast(y, dtype)
        tf_coords = tf.stack([y, x])
        tf_coords = tf.reshape(tf_coords, [2, -1])
        tf_coords = tf.transpose(tf_coords)
        
        # Map each (row, col) pair to the source image according to
        # the user-provided mapping
        tf_coords = coord_map(tf_coords.eval())
    
        # Reshape back to a (2, M, N) coordinate grid
        tf_coords = tf.constant(tf_coords)
        tf_coords = tf.transpose(tf_coords)
        tf_coords = tf.reshape(tf_coords, [-1, cols, rows])
        tf_coords = tf.transpose(tf_coords, [0, 2, 1])  
        
        # Convert from tensorflow to array for stackcopy
        coords = coords.eval()
        tf_coords = tf_coords.eval()
    
        # Place the y-coordinate mapping
        _stackcopy(coords[1, ...], tf_coords[0, ...])
        # Place the x-coordinate mapping
        _stackcopy(coords[0, ...], tf_coords[1, ...])
    
        if len(shape) == 3:
            coords[2, ...] = range(shape[2])
        return coords


# In[3]: stack copy
def _stackcopy(target, source):
    """Copy source into each color layer of target, such that::

      target[:,:,0] = target[:,:,1] = ... = source

    Parameters
    ----------
    target : (M, N) or (M, N, P) ndarray
        Target array.
    source : (M, N)
        Source array.

    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

    """
    if target.ndim == 3:
        target[:] = source[:, :, tf.newaxis]
    else:
        target[:] = source
