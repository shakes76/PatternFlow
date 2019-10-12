#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
total-variation denoising using split-Bregman optimization
"""

__author__ = "Yi-Tang Wang"
__email__ = "yitang.wang@uq.net.au"
__reference__ = ["skimage.restoration.denoise_tv_bregman"]


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
