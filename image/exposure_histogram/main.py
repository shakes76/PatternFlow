import argparse

import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.misc

from impl import histogram

def get_image(path: str) -> tf.Tensor:
    return tf.image.convert_image_dtype(tf.io.decode_image(path), tf.float32)

def get_channel_of_image(image: tf.Tensor, channel: str) -> tf.Tensor:
    """
    channel: grey | r | g | b
    """
    # Used as a lookup for the index of each channel
    colours = ['r', 'g', 'b']
    if channel == 'grey':
        return tf.image.rgb_to_grayscale(image)[:,:,0]
    else:
        try:
            index = colours.index(channel)
        except ValueError:
            raise ValueError("channel must be in: grey, r, g, b")
        return image[:,:,index]

def show_histogram(image: tf.Tensor, channel: str, nbins: int, source_range: str, normalize: bool) -> None:
    """
    Given an image tensor, display it side-by-side with it's normalized histogram
    """
    values, centers = histogram(image, nbins=nbins, source_range=source_range, normalize=normalize)
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    # Lookup table so that matplotlib displays the channel aswell
    cmap = dict(
        grey=plt.cm.gray,
        r=plt.cm.Reds,
        g=plt.cm.Greens,
        b=plt.cm.Blues
    )
    ax[0].imshow(image.eval(), cmap=cmap[channel])
    ax[0].axis('off')

    ax[1].plot(centers.eval(), values.eval(), lw=2)
    # Lookup table for nice titles
    names = dict(
        grey="grey",
        r="red",
        g="green",
        b="blue"
    )
    ax[1].set_title(f'Histogram of {names[channel]} values')

    plt.tight_layout()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Show the histogram for an image')
    parser.add_argument('image', type=str, nargs='?', help='an image path, if none is specified a raccoon face is used')
    parser.add_argument('--channel', type=str, default='grey', help='channel of the image, grey/r/g/b')
    parser.add_argument('-o', '--out', type=str, default='histogram.png', help='path of the output file')
    parser.add_argument('-n', '--nbins', type=int, default=256, help='number of bins for histogram')
    parser.add_argument('--source-range', type=str, default='image', help=(
        "'image' (default) determines the range from the input image.\n"
        "'dtype' determines the range from the expected range of the images of that data type."
    ))
    parser.add_argument('--normalize', action='store_true', default=False,
        help='normalize the histogram by the sum of its values.'
    )

    args = parser.parse_args()
    
    # Show histogram
    with tf.compat.v1.Session() as sess:
        if args.image:
            with open(args.image, 'rb') as f:
                image_tensor = get_image(f.read())
        else:
            image_tensor = tf.image.convert_image_dtype(scipy.misc.face(), tf.float32)

        image_tensor = get_channel_of_image(image_tensor, args.channel)
        show_histogram(image_tensor, channel=args.channel, nbins=args.nbins, source_range=args.source_range, normalize=args.normalize)
    
    # Save
    plt.savefig(args.out)