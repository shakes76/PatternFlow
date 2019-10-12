import histogram_matching as hmm
import matplotlib.pyplot as plt
from skimage import data
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":
    
    reference = data.coffee()
    reference1  = tf.convert_to_tensor(reference)
    image = data.astronaut()
    image1 = tf.convert_to_tensor(image)
    matched = hmm.match_histograms(image1, reference1, multichannel=True)

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