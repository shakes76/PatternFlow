# radon transform implemented in TensorFlow

import tensorflow as tf

def radon(image, theta=None, circle=True, *, preserve_range=None):
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
        if image.ndim != 2:
            raise ValueError('The input image must be 2-D')
        if theta is None:
            theta = np.arange(180)
