import matplotlib.pyplot as plt
import tensorflow as tf

from skimage import data, color
from downscale_local_mean import downscale_local_mean


def main():
    """
    get the image from the skimage library
    using the downscale_local_mean to downscaled the image
    show the original and downscaled image
    """
    #get image
    image = color.rgb2gray(data.rocket())
    
    #for the version, we can use tf.compat.v1.Session to ignore the warning 
    tf.InteractiveSession()

    #downscaled the image(if the image is ndarray)
    #image_downscaled = downscale_local_mean(image, (4, 3))

    #downscaled the image(if the image is nd tensor)
    image_tf = tf.convert_to_tensor(image)

    image_downscaled = downscale_local_mean(image_tf, (4, 3))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    #show the original and downscaled images
    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(image_downscaled, cmap='gray')
    ax[1].set_title("Downscaled image (no aliasing)")

    ax[0].set_xlim(0, 630)
    ax[0].set_ylim(400, 0)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
