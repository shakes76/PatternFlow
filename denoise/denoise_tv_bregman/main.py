#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file for testing denoise_tv_bregman function
"""

__author__ = "Yi-Tang Wang"
__email__ = "yitang.wang@uq.net.au"


import torch
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.util import random_noise

from denoise_tv_bregman import denoise_tv_bregman

original = img_as_float(data.chelsea()[100:250, 50:300])

# add noise to image
sigma = 0.155
noisy = random_noise(original, var=sigma**2)

# the denoised image is returned as torch tensor
# use .numpy() to convert into numpy array for plotting
denoised = denoise_tv_bregman(torch.FloatTensor(noisy), weight=0.1).numpy()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('Noisy')
ax[1].imshow(denoised)
ax[1].axis('off')
ax[1].set_title('Denoised')

plt.show()
