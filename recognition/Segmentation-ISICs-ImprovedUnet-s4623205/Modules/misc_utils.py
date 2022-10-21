"""
Misc functions to use in the project

@author: Jeng-Chung Lien
@student id: 46232050
@email: jengchung.lien@uqconnect.edu.au
"""
import sys
from math import ceil, pow


def progressbar(current, max_size):
    """
    Function for displaying the progress in the console

    Parameters
    ----------
    current : integer
      The current progress count
    max_size : integer
      The maximum progress count
    """
    sys.stdout.write('\r')
    progress = ceil((100 / int(max_size)) * current)
    sys.stdout.write("[%-100s] %d%%" % ('=' * progress, progress))
    sys.stdout.flush()


def get_close2power(value):
    """
    Function to get the value closest of value power of count smaller than value

    Parameters
    ----------
    value : integer
      The parameter to get the value closest to

    Returns
    -------
    result : integer
      An integer value of value power of count
    """
    result = 0
    count = 0

    while result <= value:
        temp_result = pow(2, count)
        count += 1
        if temp_result <= value:
            result = temp_result
        else:
            break

    return int(result)


def image_mask_combiner(image, mask):
    """
    Function to combine the image and mask, showing the mask as blue area on the image

    Parameters
    ----------
    image : float32 numpy array or tensors
      The image shape (h, w, 3) is required. Values must be between [0, 1].
    mask : numpy array or tensors
      The mask shape (h, w) is required. Values must be between [0, 1].

    Returns
    -------
    image : float32 numpy array or tensors
      The combined image with mask area as color blue
    """
    image[:, :, 2] = (255 * image[:, :, 2] * (1 - mask) + (255 * mask * mask)) / 255

    return image
