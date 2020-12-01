from scipy.signal import convolve2d as conv2
import numpy as np
import matplotlib.pyplot as plt
from unspvd_wiener import unsupervised_wiener
from skimage import color
from PIL import Image


def main():
    """Test script for unsupervised_wiener
    """
    img = color.rgb2gray(np.asarray(Image.open("resources/chelsea.png")))
    psf = np.ones((5, 5)) / 25
    noised_img = conv2(img, psf, 'same')
    noised_img += 0.7 * noised_img.std() * np.random.standard_normal(noised_img.shape)
    deconvolved, _ = unsupervised_wiener(noised_img, psf)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5),
                           sharex=True, sharey=True)
    plt.gray()
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(noised_img, vmin=deconvolved.min(), vmax=deconvolved.max())
    ax[1].axis('off')
    ax[1].set_title('Noised')
    ax[2].imshow(deconvolved)
    ax[2].axis('off')
    ax[2].set_title('Denoised')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
