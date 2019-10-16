import os
import numpy as np
import itertools

from skimage import data_dir
from skimage.io import imread
from skimage.transform import radon, iradon_sart, rescale

from skimage._shared import testing
from skimage._shared.testing import test_parallel
from skimage._shared._warnings import expected_warnings

from iradon import iradon
PHANTOM = imread(os.path.join(data_dir, "phantom.png"),
                 as_gray=True)[::2, ::2]
PHANTOM = rescale(PHANTOM, 0.5, order=1,
                  mode='constant', anti_aliasing=False, multichannel=False)

filter_types = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
interpolation_types = ['linear', 'nearest']
radon_iradon_inputs = list(itertools.product(interpolation_types,
                                             filter_types))
# cubic interpolation is slow; only run one test for it
radon_iradon_inputs.append(('cubic', 'shepp-logan'))

@testing.parametrize("interpolation_type, filter_type",
                     radon_iradon_inputs)

def test_radon_iradon(interpolation_type, filter_type):
    check_radon_iradon(interpolation_type, filter_type)

test_radon_iradon("linear", "ramp")
    
def check_radon_iradon(interpolation_type, filter_type):
    debug = False
    image = PHANTOM
    reconstructed = iradon(radon(image, circle=False), filter=filter_type,
                           interpolation=interpolation_type, circle=False)
    delta = np.mean(np.abs(image - reconstructed))
    print('\n\tmean error:', delta)
    if debug:
        _debug_plot(image, reconstructed)
    if filter_type in ('ramp', 'shepp-logan'):
        if interpolation_type == 'nearest':
            allowed_delta = 0.03
        else:
            allowed_delta = 0.025
    else:
        allowed_delta = 0.05
    assert delta < allowed_delta

def _debug_plot(original, result, sinogram=None):
    from matplotlib import pyplot as plt
    imkwargs = dict(cmap='gray', interpolation='nearest')
    if sinogram is None:
        plt.figure(figsize=(15, 6))
        sp = 130
    else:
        plt.figure(figsize=(11, 11))
        sp = 221
        plt.subplot(sp + 0)
        plt.imshow(sinogram, aspect='auto', **imkwargs)
    plt.subplot(sp + 1)
    plt.imshow(original, **imkwargs)
    plt.subplot(sp + 2)
    plt.imshow(result, vmin=original.min(), vmax=original.max(), **imkwargs)
    plt.subplot(sp + 3)
    plt.imshow(result - original, **imkwargs)
    plt.colorbar()
    plt.show()
