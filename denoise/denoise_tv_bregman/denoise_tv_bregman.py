#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
total-variation denoising using split-Bregman optimization

The denoised image will be returned as type torch.Tensor
Should be convert to numpy array for plotting
"""

__author__ = "Yi-Tang Wang"
__email__ = "yitang.wang@uq.net.au"
__reference__ = ["skimage.restoration.denoise_tv_bregman"]

import torch
import math


def denoise_tv_bregman(image, weight, max_iter=100, eps=1e-3):
    """Perform total-variation denoising using split-Bregman optimization.

    Parameters:
        image (torch.Tensor):
            Input data to be denoised.
        weight (float):
            Denoising weight. The smaller the `weight`, the more denoising (at
            the expense of less similarity to the `input`).
        eps (float):
            Optional
            The threshold of distance between denoised image in iterations
            The algorithm stops when image distance is smaller than eps
        max_iter (int):
            Optional
            Maximal number of iterations used for the optimization.

    Returns:
        out (torch.Tensor): denoised image
    """
    image = atleast_3d(image)

    img_shape = list(image.shape)
    rows = img_shape[0]
    rows2 = rows + 2
    cols = img_shape[1]
    cols2 = cols + 2
    dims = img_shape[2]
    total = rows * cols * dims
    shape_extend = (rows2, cols2, dims)
    # out is zeros-like tensor with size as shape_extend
    out = torch.zeros(shape_extend, dtype=torch.double)

    dx = out.clone().detach()
    dy = out.clone().detach()
    bx = out.clone().detach()
    by = out.clone().detach()

    lam = 2 * weight
    rmse = float("inf")
    norm = (weight + 4 * lam)

    out_rows, out_cols = out.shape[:2]
    out[1:out_rows-1, 1:out_cols-1] = image

    out = fill_extend(image, out)

    i = 0
    # iterative optimization method
    # the Gauss-Seidel method
    while i < max_iter and rmse > eps:

        rmse = 0

        # doing pixel by pixel
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                for k in range(dims):

                    uprev = out[r, c, k]

                    ux = out[r, c + 1, k] - uprev
                    uy = out[r + 1, c, k] - uprev

                    # the regularization term
                    # to keep the denoised image more like original
                    regularization = weight * image[r - 1, c - 1, k]

                    unew = (
                        lam * (
                            out[r + 1, c, k]
                            + out[r - 1, c, k]
                            + out[r, c + 1, k]
                            + out[r, c - 1, k]

                            + dx[r, c - 1, k]
                            - dx[r, c, k]
                            + dy[r - 1, c, k]
                            - dy[r, c, k]

                            - bx[r, c - 1, k]
                            + bx[r, c, k]
                            - by[r - 1, c, k]
                            + by[r, c, k]
                        ) + regularization
                    ) / norm
                    out[r, c, k] = unew

                    # sum up ||u-f||^2
                    rmse += (unew - uprev)**2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    tx = ux + bxx
                    ty = uy + byy
                    s = math.sqrt(tx * tx + ty * ty)
                    dxx = s * lam * tx / (s * lam + 1)
                    dyy = s * lam * ty / (s * lam + 1)

        rmse = math.sqrt(rmse / total)
        i += 1

    # return the denoised image excluding the extended area
    return out[1:-1, 1:-1]


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


def fill_extend(image, out):
    """fill the extended area in out img with original img"""
    out_rows, out_cols = out.shape[:2]
    rows, cols = out_rows - 2, out_cols - 2
    out[0, 1:out_cols-1] = image[1, :]
    out[1:out_rows-1, 0] = image[:, 1]
    out[out_rows-1, 1:out_cols-1] = image[rows-1, :]
    out[1:out_rows-1, out_cols-1] = image[:, cols-1]
    return out
