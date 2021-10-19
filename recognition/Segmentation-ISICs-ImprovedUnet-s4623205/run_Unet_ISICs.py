import matplotlib.pyplot as plt
import cv2
import sys
from math import ceil
import glob
import numpy as np


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


def get_min_imageshape(path):
    """
    Function to get the minimum image shape

    Parameters
    ----------
    path : string
      Directory of where the target images are

    Returns
    -------
    min_shape : list
      A list of the minimum image shape [height, width]
    """
    img_paths = sorted(glob.glob(path))
    length = len(img_paths)
    count = 0
    image_shapes = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_shape = img.shape
        shape_info = [img_shape[0]*img_shape[1], [img_shape[0], img_shape[1]]]
        image_shapes.append(shape_info)
        progressbar(count, length)
        count += 1

    image_shapes = np.array(image_shapes, dtype=object)
    index = np.where(image_shapes[:, 0] == min(image_shapes[:, 0]))
    min_shape = image_shapes[index][0][1]

    return min_shape


def load_rgbimages(path, height, width):
    """
    Function to load and preprocess rgb images from path to memory

    Parameters
    ----------
    path : string
      Directory of where the target images are
    height : integer
      Parameter to resize the image height
    width : integer
      Parameter to resize the image width

    Returns
    -------
    images : float32 numpy array
      A data type float32 numpy array of the preprocessed images
    """
    img_paths = sorted(glob.glob(path))
    length = len(img_paths)
    count = 0
    images = []

    for img_path in img_paths:
        # OpenCV reads images as BGR, convert it to RGB when reading
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        images.append(img)
        progressbar(count, length)
        count += 1

    images = np.array(images, np.float32)
    images = images / 255.

    return images


def main():
    image_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"

    print("Getting minimum image shape...")
    # Image shapes are not consistent, get the minimum image shape. Shape of [283, 340] in this case.
    min_img_shape = get_min_imageshape(image_path)
    img_height = min_img_shape[0]
    img_width = min_img_shape[1]
    print("\nMin Image Height:", img_height)
    print("Min Image Width:", img_width)

    print("\nLoad and preprocess RGB images...")
    images = load_rgbimages(image_path, img_height, img_width)
    print("\nPlotting the first preprocessed RGB image...")
    plt.imshow(images[0])
    plt.show()


if __name__ == "__main__":
    main()