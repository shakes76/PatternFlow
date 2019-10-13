#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:21:05 2019

@author: s4503302
"""
import numpy as np
from skimage import img_as_float
import os
#%%
import matplotlib.pyplot as plt
import matplotlib.image as img
document = os.path.join(os.path.expanduser("~"), "/Users/s4503302/Documents/LLD_DCGAN")
loadPath_2x = os.path.join(document, "hat.png")
image = img.imread(loadPath_2x)
img_test=image.copy()
plt.imshow(img_test)
theHat=image.copy()

plt.imshow(theHat)
theHat_denoise = denoise_tv_chambolle(theHat)
plt.imshow(theHat_denoise)


ttt = torch.from_numpy(image)
        
#%%
def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : ndarray
        Denoised array of floats.
    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    ndim = image.ndim
    p = np.zeros((image.ndim, ) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = np.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out

#%%
def denoise_tv_chambolle(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : ndarray of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.
    Returns
    -------
    out : ndarray
        Denoised image.
    Notes
    -----
    Make sure to set the multichannel parameter appropriately for color images.
    The principle of total variation denoising is explained in
    https://en.wikipedia.org/wiki/Total_variation_denoising
    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.
    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.
    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.
    Examples
    --------
    2D example on astronaut image:
    >>> from skimage import color, data
    >>> img = color.rgb2gray(data.astronaut())[:50, :50]
    >>> img += 0.5 * img.std() * np.random.randn(*img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)
    3D example on synthetic data:
    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(np.float)
    >>> mask += 0.2*np.random.randn(*mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)
    """

    im_type = image.dtype
    if not im_type.kind == 'f':
        image = img_as_float(image)

    if multichannel:
        out = np.zeros_like(image)
        for c in range(image.shape[-1]):
            out[..., c] = _denoise_tv_chambolle_nd(image[..., c], weight, eps,
                                                   n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, n_iter_max)
    return out