#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:08:43 2019
Last edited on 06/ Nov/ 2019

author: Wei-Chung

description: this is the test function for the denoise function "denoise_tv_chambolle"
in torch version. It supports data structure "torch.tensor". 

"""


import torch
import unittest
from skimage import data, img_as_float, exposure, img_as_ubyte
from skimage import restoration, data, color, img_as_float, measure
import sys
sys.path.append('/PatternFlow/45033027/')
from torch_denoise_tv_chambolle import denoise_tv_chambolle_torch
from skimage._shared.testing import (assert_equal, assert_almost_equal,
                                     assert_warns, assert_)
import torchvision.transforms.functional as F  
from torchvision import transforms
import numpy as np 
import matplotlib.pyplot as plt


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
    assert_((denoised_int_astroT.numpy()).dtype == np.float)
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
    assert_((resT.numpy()).dtype == np.float)
    assert_(resT.std() * 255 < mask.std())
    
def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100*np.sin(np.linspace(0, 8*np.pi, 1000))
    x += 20 * np.random.rand(x.size)
    x = np.clip(x, 0, 255)
    xT = torch.tensor(x)
    resT = denoise_tv_chambolle_torch(xT.type(torch.uint8), weight=0.1)    
    assert_((resT.numpy()).dtype == np.float)
    assert_(resT.std() * 255 < x.std())

def test_denoise_tv_chambolle_4d():
    """ TV denoising for a 4D input."""
    im = 255 * np.random.rand(8, 8, 8, 8)
    imT = torch.tensor(im)
    resT = denoise_tv_chambolle_torch(imT.type(torch.uint8), weight=0.1)
    assert_((resT.numpy()).dtype == np.float)
    assert_(resT.std() * 255 < im.std())
    
def test_denoise_tv_chambolle_weighting():
    # make sure a specified weight gives consistent results regardless of
    # the number of input image dimensions
    rstate = np.random.RandomState(1234)
    img2d = astro_gray.copy()
    img2d += 0.15 * rstate.standard_normal(img2d.shape)
    img2d = np.clip(img2d, 0, 1)

    # generate 4D image by tiling
    img4d = np.tile(img2d[..., None, None], (1, 1, 2, 2))
    img2dT = torch.tensor(img2d)
    img4dT = torch.tensor(img4d)
    w = 0.2
    denoised_2d = denoise_tv_chambolle_torch(img2dT, weight=w)
    denoised_4d = denoise_tv_chambolle_torch(img4dT, weight=w)
    assert_(measure.compare_ssim(denoised_2d.numpy(),
                                denoised_4d[:, :, 0, 0].numpy()) > 0.99)
    
def test_denoise_tv_chambolle_multichannel():
    denoised0 = denoise_tv_chambolle_torch(astroT[..., 0], weight=0.1)
    denoised = denoise_tv_chambolle_torch(astroT, weight=0.1,
                                                multichannel=True)
    assert_equal(denoised[..., 0].numpy(), denoised0.numpy())

    # tile astronaut subset to generate 3D+channels data
    astro3 = np.tile(astro[:64, :64, np.newaxis, :], [1, 1, 2, 1])
    # modify along tiled dimension to give non-zero gradient on 3rd axis
    astro3[:, :, 0, :] = 2*astro3[:, :, 0, :]
    astro3T = torch.tensor(astro3)
    denoised0 = denoise_tv_chambolle_torch(astro3T[..., 0], weight=0.1)
    denoised = denoise_tv_chambolle_torch(astro3T, weight=0.1,
                                                multichannel=True)
    assert_equal(denoised[..., 0].numpy(), denoised0.numpy())

if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    astro = img_as_float(data.astronaut()[:128, :128])
    astro_gray = color.rgb2gray(astro)
    astroT = torch.tensor(astro,dtype = torch.float32)
    astro_grayT = torch.tensor(astro_gray,dtype = torch.float32)
    checkerboard = img_as_float(data.checkerboard())
    checkerboard_gray = color.gray2rgb(checkerboard)
    astcheckerboardroT = torch.tensor(checkerboard,dtype = torch.float32)
    checkerboard_grayT = torch.tensor(checkerboard_gray,dtype = torch.float32)
    
    #test the all test cases from skimage package
    test_denoise_tv_chambolle_2d()
    test_denoise_tv_chambolle_multichannel()
    test_denoise_tv_chambolle_float_result_range()
    test_denoise_tv_chambolle_3d()
    test_denoise_tv_chambolle_1d()
    test_denoise_tv_chambolle_4d()
    test_denoise_tv_chambolle_weighting()
    
        
    coffee = img_as_float(data.coffee())
    coffeeT = torch.tensor(coffee)
    #add noise to the original image to test the denoising effect
    noise  = torch.randn(coffeeT.size(),dtype = coffeeT.dtype)*0.15
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8,8))
    ax1.imshow(coffee+noise.numpy())
    ax1.set_title('Original Image')
    #result of torch
    ax2.imshow(denoise_tv_chambolle_torch(coffeeT+noise))
    ax2.set_title('Denoised(torch)')
    #result of numpy
    ax3.imshow(restoration.denoise_tv_chambolle(coffee+noise.numpy()))
    ax3.set_title('Denoised(numpy)')
    plt.show()    
        