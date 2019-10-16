import argparse

import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.misc

from impl import histogram

def get_image(path: str) -> tf.Tensor:
    return tf.io.decode_image(path)

def get_channel_of_image(image: tf.Tensor, channel: str) -> tf.Tensor:
    """
    channel: grey | r | g | b
    """
    colours = ['r', 'g', 'b']
    if channel == 'grey':
        return tf.image.rgb_to_grayscale(image)[:,:,0]
    else:
        try:
            index = colours.index(channel)
        except ValueError:
            raise ValueError("channel must be in: grey, r, g, b")
        return image[:,:,index]

def show_histogram(image: tf.Tensor) -> None:
    """
    Given an image tensor, display it side-by-side with it's normalized histogram
    """
    values, centers = histogram(image, normalize=True)
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].imshow(image.eval(), cmap=plt.cm.gray)
    ax[0].axis('off')

    ax[1].plot(centers.eval(), values.eval(), lw=2)
    ax[1].set_title('Histogram of grey values')

    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show the histogram for an image')
    parser.add_argument('image', type=str, nargs='?', help='an image path')
    parser.add_argument('--channel', type=str, default='grey', help='channel of the image, grey/r/g/b')
    parser.add_argument('-o', '--out', type=str, default='histogram.png', help='path of the output file')

    args = parser.parse_args()
    
    with tf.compat.v1.Session() as sess:
        if args.image:
            image_tensor = get_image(args.image)
        else:
            image_tensor = scipy.misc.face()

        image_tensor = get_channel_of_image(image_tensor, args.channel)
        show_histogram(image_tensor)
    
    plt.savefig(args.out)