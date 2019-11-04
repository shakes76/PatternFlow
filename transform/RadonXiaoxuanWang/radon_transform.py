import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import _warps as wp
from skimage._shared.utils import convert_to_float

def radon(image, theta=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.
    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.
    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989
    Notes
    -----
    Based on code of Justin K. Romberg
    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)
    """
    
    image = convert_to_float(image, None)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    with tf.compat.v1.Session() as sess:
        diagonal = tf.sqrt(2.0) * max(image.shape)
        pad = [tf.dtypes.cast(tf.math.ceil(diagonal - s), tf.int32) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = tf.pad(image, pad_width, mode='constant', constant_values=0)

        # padded_image is always square
        if padded_image.shape[0] != padded_image.shape[1]:
            raise ValueError('padded_image must be a square')

        center = padded_image.shape[0] // 2

        radon_image = np.zeros((padded_image.shape[0], len(theta)))

        # convert degree to radian

        for i, angle in enumerate(np.deg2rad(theta)):
            cos_a, sin_a = tf.math.cos(angle), tf.math.sin(angle)

            R = tf.Variable(list([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                          [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                          [0, 0, 1]]))

            init_op = tf.compat.v1.global_variables_initializer()

            sess.run(init_op)

            rotated = wp.warp(padded_image.eval(), R.eval(), clip=False)

            # update transformed image
            radon_image[:, i] = rotated.sum(0)

    return radon_image
