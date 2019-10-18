import matplotlib.pyplot as plt
import match_histograms as hist
import tensorflow as tf
from skimage import data

def cumulative(image, reference):
    image = tf.convert_to_tensor(image)
    reference = tf.convert_to_tensor(reference)

    matched = hist.match_histograms(image, reference, multichannel=True)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(image)
    ax1.set_title('Source')
    ax2.imshow(reference)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    reference = data.coffee()
    source = data.astronaut()

    cumulative(source,reference)