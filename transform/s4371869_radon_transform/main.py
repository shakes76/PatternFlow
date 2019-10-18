# Standard imports
import os
import time
import pstats
import cProfile
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon, rescale

try:
    os.chdir(os.path.join("transform", "s4371869_radon_transform"))
except OSError:
    pass

# Custom imports
from radon_transform import radon as my_radon

def main():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.2, mode='reflect', multichannel=False)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4.5, 12))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 180., max(image.shape) * 1.5, endpoint=False)
    tic = time.time()
    sinogram = radon(image, theta=theta, circle=True)
    toc = time.time()
    ax2.set_title("Scikit Radon transform\n(t={0:.3f})".format(toc - tic))
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    pr = cProfile.Profile()
    # pr.enable()
    # tic = time.time()
    my_sinogram = my_radon(image, theta=theta, circle=True)
    toc = time.time()
    # pr.disable()
    # pr.print_stats(sort='cumtime')
    ax3.set_title("My Radon transform\n(t={0:.3f})".format(toc - tic))
    ax3.set_xlabel("Projection angle (deg)")
    ax3.set_ylabel("Projection position (pixels)")
    ax3.imshow(my_sinogram, cmap=plt.cm.Greys_r,
            extent=(0, 180, 0, my_sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()

    reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
    error = reconstruction_fbp - image
    print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}")

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    plt.show()


if __name__ == '__main__':
    main()