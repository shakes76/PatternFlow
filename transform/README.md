# Transforms: _**warp_coords**_
----------
Ported the [skimage.transform.warp_coords](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp_coords) to Tensorflow

**Description**
----------
**warp_coords(coord_map, shape, dtype=<class 'tensorflow.float64'>)** 

This function build a source coordinates for the output of a 2-D image warp.

**Parameters**

`coord_map` : callable like GeometricTransform.inverse
    Return input coordinates for given output coordinates.
    Coordinates are in the shape (P, 2), where P is the number
    of coordinates and each element is a ``(row, col)`` pair.
    
`shape` : tuple
    Shape of output image `(rows, cols[, bands])`.
    
`dtype` : tensorflow.dtype or string
    dtype for return value (sane choices: float32 or float64).

**Returns**

`coords` : (ndim, rows, cols[, bands]) array of dtype `dtype`
        Coordinates for `scipy.ndimage.map_coordinates`, that will yield
        an image of shape (orows, ocols, bands) by drawing from source
        points according to the `coord_transform_fn`.

**Notes**

This is a lower-level routine that produces the source coordinates for 2-D
images used by `warp()`.

It is provided separately from `warp` to give additional flexibility to
users who would like, for example, to re-use a particular coordinate
mapping, to use specific dtypes at various points along the the
image-warping process, or to implement different post-processing logic
than `warp` performs after the call to `ndi.map_coordinates`.

**Examples**

```
def test_warp_coords_example():
    image = data.astronaut().astype(np.float32)
    coords = warp_coords(shift_down10_left20, image.shape, dtype=tf.float32)
    warped_image = map_coordinates(image, coords)
    
    plot_result(image, warped_image)
```
``` 
def shift_down10_left20(xy):
    return xy - np.array([-20, 10])[None, :]
```
```    
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
```

**Dependencies**
----------
``_stackcopy(target, source)``
Copy source into each color layer of target, such that::

target[:,:,0] = target[:,:,1] = ... = source

**Parameters**

`target` : (M, N) or (M, N, P) ndarray
    Target array.
    
`source` : (M, N)
    Source array.

**Notes**

Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

