#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
total-variation denoising using split-Bregman optimization
"""

__author__ = "Yi-Tang Wang"
__email__ = "yitang.wang@uq.net.au"
__reference__ = ["skimage.restoration.denoise_tv_bregman"]

import torch


def denoise_tv_bregman(image, weight, max_iter=100, eps=1e-3, isotropic=True):
    """Perform total-variation denoising using split-Bregman optimization.

    Parameters:
        image (torch.Tensor):
            Input data to be denoised.
        weight : float
            Denoising weight. The smaller the `weight`, the more denoising (at
            the expense of less similarity to the `input`). The regularization
            parameter `lambda` is chosen as `2 * weight`.
        eps : float, optional
            The stop criterion of the algorithm
            The algorithm stops when RMSE is smaller than eps

        max_iter : int, optional
            Maximal number of iterations used for the optimization.
        isotropic : boolean, optional
            Switch between isotropic and anisotropic TV denoising.

    Returns:
        u : denoised image
    """
    pass


def atleast_3d(image):
    """to ensure the image has at least 3 dimensions

    if the input image already has at least 3 dimensions, just return the image
    otherwise, extend the dimensionality of the image to 3 dimensions

    Parameters:
        image (torch.Tensor):
            input image

    Return:
        image :
            image that at least 3 dimensions
    """
    dim = list(image.shape)

    if len(dim) >= 3:
        return image
    else:
        dim.append(1)
        return image.view(dim)


def reflect_image(image):
    """reflect the input image and extend with paddings"""
    pass
