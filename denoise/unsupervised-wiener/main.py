from scipy.signal import convolve2d as conv2
import numpy as np
import matplotlib.pyplot as plt
from unspvd_wiener import unsupervised_wiener
from skimage import color
from PIL import Image


def main():
    astro = color.rgb2gray(np.asarray(Image.open("resources/astronaut.jpg")))
    psf = np.ones((5, 5)) / 25
    astro = conv2(astro, psf, 'same')
    astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)
    deconvolved, _ = unsupervised_wiener(astro, psf)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                           sharex=True, sharey=True)
    plt.gray()
    ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
    ax[0].axis('off')
    ax[0].set_title('Data')
    ax[1].imshow(deconvolved)
    ax[1].axis('off')
    ax[1].set_title('Self tuned restoration')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
