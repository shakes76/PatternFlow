# File containing the data loader for loading and preprocessing your data
import os
from os import listdir

import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt, patches

IMAGE_SIZE = 640

VALIDATION_DATA_PATH = "Datasets/Validation/validation_data/"
TESTING_DATA_PATH = "Datasets/Testing/testing_data/"
TRAINING_DATA_PATH = "Datasets/Training/training_data/"

VALIDATION_MASK_PATH = "Datasets/Validation/validation_mask/"
TESTING_MASK_PATH = "Datasets/Testing/testing_mask/"
TRAINING_MASK_PATH = "Datasets/Training/training_mask/"


def load_resize_images(image_path):
    file_names = listdir(image_path)

    for filename in file_names:
        if filename.endswith(".DS_Store"):
            continue

        with im.open(image_path + filename) as img:
            width, height = img.size

            # Change the shape of the images so all the images have a maximum axis
            # of 640 and keep the aspect ratio
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            # Save image in the same
            img.save(image_path + filename)

        # Another function to open mask data and call the bounding box function
        print(img.size)

        new_images(image_path, filename)


def new_images(mask_directory, filename):
    # Name of the file without jpg ending
    name = filename.replace('.png', '')

    print("File name: " + name)

    # Open mask directory images
    with im.open(mask_directory + filename) as img:
        print(img.size)

        # Need image in a numpy array
        image_array = np.array(img)

        # Create figure and axes
        fig, ax = plt.subplots()

        ax.imshow(img)

        bounding_box_info = generate_bounding_box(image_array)

        print(bounding_box_info)

        # Create a Rectangle patch
        rect = patches.Rectangle((bounding_box_info[0], bounding_box_info[1]), bounding_box_info[2],
                                 bounding_box_info[3],
                                 linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()

        return


def generate_bounding_box(image):
    print(image.shape)

    # Compute the xMin, xMax, yMin, yMax
    # Average xMin and xMax  and yMin and yMax to compute the centre  of the bounding box
    # Divide the x average by the width of the image and the y average by the height of the image

    height, width = image.shape

    # argwhere funds the indices of array elements that are non-zero, grouped by element
    mask_y_indices = np.sort(np.argwhere(image > 1)[:, 0])
    print("Mask y indices: " + mask_y_indices)

    mask_x_indices = np.sort(np.argwhere(image > 1)[:, 1])
    print("Mask x indices: " + mask_x_indices)

    # X- axis
    x_min = mask_x_indices[0]
    print("X min: " + x_min)

    x_max = mask_x_indices[-1]
    print("x max: " + x_max)

    # Y- axis
    y_min = mask_y_indices[-1]
    print("y min: " + y_min)

    # The top left is 0
    y_max = mask_y_indices[0]
    print("y max: " + y_max)

    # Normalise
    x_avg_normalised = ((x_min + x_max) / 2) / width
    print("x avg norm: " + x_avg_normalised)

    y_avg_normalised = ((y_min + y_max) / 2) / height
    print("y avg norm: " + y_avg_normalised)


    # Bounding Box dimensions
    width_box = (x_max - x_min) / width
    height_box = (y_max - y_min) / height
    print("box width: " + width_box)
    print("box height: " + height_box)

    return x_avg_normalised, y_avg_normalised, width_box, height_box


def clean_up_directory(image_directory):
    print(image_directory)
    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            os.remove(image_directory + filename)


def main():
    # Do this for all three directories
    # Not necessary for mask data
    # clean_up_directory(VALIDATION_DATA_PATH)
    # clean_up_directory(TESTING_DATA_PATH)
    # clean_up_directory(TRAINING_DATA_PATH)

    # Resize all images
    load_resize_images(VALIDATION_MASK_PATH)

    # load_resize_images(TESTING_DATA_PATH)
    # load_resize_images(TRAINING_DATA_PATH)

    load_resize_images(VALIDATION_MASK_PATH)

    # One hot encode into two categories


if __name__ == '__main__':
    main()
