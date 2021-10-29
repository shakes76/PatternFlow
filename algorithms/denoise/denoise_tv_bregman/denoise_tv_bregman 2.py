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
            Denoising weight. The smaller the 'weight', the more denoising (at
            the expense of less similarity to the 'input').
        max_iter (int):
            Optional
            Maximal number of iterations used for the optimization.
        eps (float):
            Optional
            The threshold of distance between denoised image in iterations
            The algorithm stops when image distance is smaller than eps

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
    # out is firstly created as zeros-like tensor with size as shape_extend
    out = torch.zeros(shape_extend, dtype=torch.float)

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
    regularization = torch.mul(image, weight)
    # iterative optimization method
    # split-Bregman iteration
    while i < max_iter and rmse > eps:
        uprev = out[1:-1, 1:-1, :]

        ux = out[1:-1, 2:, :] - uprev
        uy = out[2:, 1:-1, :] - uprev

        unew = torch.div(
            (torch.mul((out[2:, 1:-1, :]
                    + out[0:-2, 1:-1, :]
                    + out[1:-1, 2:, :]
                    + out[1:-1, 0:-2, :]

                    + dx[1:-1, 0:-2, :]
                    - dx[1:-1, 1:-1, :]
                    + dy[0:-2, 1:-1, :]
                    - dy[1:-1, 1:-1, :]

                    - bx[1:-1, 0:-2, :]
                    + bx[1:-1, 1:-1, :]
                    - by[0:-2, 1:-1, :]
                    + by[1:-1, 1:-1, :]), lam) + regularization),
             norm)
        out[1:-1, 1:-1, :] = unew.clone().detach()

        rmse = torch.norm(unew-uprev, p=2)

        bxx = bx[1:-1, 1:-1, :].clone().detach()
        byy = by[1:-1, 1:-1, :].clone().detach()

        tx = ux + bxx
        ty = uy + byy
        s = torch.sqrt(torch.pow(tx, 2)+torch.pow(ty, 2))
        dxx = torch.div(torch.addcmul(torch.zeros(s.shape, dtype=torch.float), lam, s, tx),
                        torch.add(torch.mul(s, lam), 1))
        dyy = torch.div(torch.addcmul(torch.zeros(s.shape, dtype=torch.float), lam, s, ty),
                        torch.add(torch.mul(s, lam), 1))

        dx[1:-1, 1:-1, :] = dxx.clone().detach()
        dy[1:-1, 1:-1, :] = dyy.clone().detach()

        bx[1:-1, 1:-1, :] += ux - dxx
        by[1:-1, 1:-1, :] += uy - dyy

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
        image (torch.Tensor):
            image that has at least 3 dimensions
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
