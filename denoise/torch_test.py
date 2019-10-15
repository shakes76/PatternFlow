#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:08:43 2019

@author: s4503302
"""


import torch
import unittest
from skimage import data, img_as_float, exposure, img_as_ubyte
from skimage import restoration, data, color, img_as_float, measure


import sys
sys.path.append('/PatternFlow/denoise/')
from torch_denoise_tv_chambolle import denoise_tv_chambolle_torch
from skimage._shared.testing import (assert_equal, assert_almost_equal,
                                     assert_warns, assert_)
import torchvision.transforms.functional as F  
from torchvision import transforms
import numpy as np 


#%% 

def test_denoise_tv_chambolle_2d():
    # astronaut image
    img = astro_grayT
    # add noise to astronaut
    img = img + 0.5 * img.std() * torch.rand(*img.shape)
    # clamp noise so that it does not exceed allowed range for float images.
    img = torch.clamp(img, 0, 1)
    # denoise
    denoised_astro = denoise_tv_chambolle_torch(img, weight=0.1)
    # which dtype?
    assert_(denoised_astro.dtype in [torch.float, torch.float32, torch.float64])
    from scipy import ndimage as ndi
    grad = ndi.morphological_gradient(img, size=((3, 3)))
    grad_denoised = ndi.morphological_gradient(denoised_astro, size=((3, 3)))
    # test if the total variation has decreased
    assert_(grad_denoised.dtype == np.float32)
    assert_(np.sqrt((grad_denoised**2).sum()) < np.sqrt((grad**2).sum()))

def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_grayT
    int_astroT = torch.mul(img, 255).type(torch.uint8)
    assert_(torch.max(int_astroT) > 1)
    denoised_int_astroT = denoise_tv_chambolle_torch(int_astroT, weight=0.1)
    # test if the value range of output float data is within [0.0:1.0]
    assert_(denoised_int_astroT.dtype == torch.float)
    assert_(torch.max(denoised_int_astroT) <= 1.0)
    assert_(torch.min(denoised_int_astroT) >= 0.0)

def test_denoise_tv_chambolle_3d():
    """Apply the TV denoising algorithm on a 3D image representing a sphere."""
    x, y, z = np.ogrid[0:40, 0:40, 0:40]
    mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    mask = 100 * mask.astype(np.float)
    mask += 60
    mask += 20 * np.random.rand(*mask.shape)
    mask[mask < 0] = 0
    mask[mask > 255] = 255
    maskT= torch.tensor(mask,dtype = torch.float32)
    resT = denoise_tv_chambolle_torch(maskT.type(torch.uint8), weight=0.1)
    assert_(resT.dtype == torch.float)
    assert_(resT.std() * 255 < mask.std())