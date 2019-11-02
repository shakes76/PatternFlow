import matplotlib.pyplot as plt
from skimage import data, color
from downscale_local_mean import downscale_local_mean


def main():

    image = color.rgb2gray(data.coffee())
    image_downscaled = downscale_local_mean(image, (4, 3))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(image_downscaled, cmap='gray')
    ax[1].set_title("Downscaled image (no aliasing)")

    ax[0].set_xlim(0, 600)
    ax[0].set_ylim(400, 0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
