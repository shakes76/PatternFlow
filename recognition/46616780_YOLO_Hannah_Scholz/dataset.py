# File containing the data loader for loading and preprocessing your data
import os
from os import listdir

import numpy as np
from PIL import Image as im

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
            if width > height:
                img = img.resize((IMAGE_SIZE, round(height * (IMAGE_SIZE / width))))
            else:
                img = img.resize((round(width * (IMAGE_SIZE / height)), IMAGE_SIZE))

        # Another function to open mask data and call the bounding box function
        print(img.size)

        new_images(image_path, filename)


def new_images(mask_directory, filename):

    masks = {}
    # Name of the file without jpg ending
    name = filename.replace('.png', '')

    print(name)


    # Open mask directory images
    with im.open(mask_directory + filename) as img:
        masks['base'] = img
        keys = list(masks.keys())

        for key in keys:
            new_mask = masks[key]
            # Need image in a numpy array
            image_array = np.array(new_mask)

            bounding_box_info = generate_bounding_box(image_array)
            print(bounding_box_info)


def generate_bounding_box(image):
    print(image.shape)

    # Compute the xMin, xMax, yMin, yMax
    # Average xMin and xMax  and yMin and yMax to compute the centre  of the bounding box
    # Divide the x average by the width of the image and the y average by the height of the image

    height, width = image.shape

    # argwhere funds the indices of array elements that are non-zero, grouped by element
    mask_y_indices = np.sort(np.argwhere(image > 1)[:, 0])
    mask_x_indices = np.sort(np.argwhere(image > 1)[:, 1])

    # X- axis
    x_min = mask_x_indices[0]
    x_max = mask_x_indices[-1]

    # Y- axis
    y_min = mask_y_indices[-1]
    # The top left is 0
    y_max = mask_y_indices[0]

    # Normalise
    x_avg_normalised = ((x_min + x_max) / 2) / width
    y_avg_normalised = ((y_min + y_max) / 2) / height

    # Bounding Box dimensions
    width_box = (x_max - x_min) / width
    height_box = (y_max - y_min) / height

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
    #
    # # Resize all images
    # load_resize_images(VALIDATION_DATA_PATH)
    # load_resize_images(TESTING_DATA_PATH)
    # load_resize_images(TRAINING_DATA_PATH)

    load_resize_images(VALIDATION_MASK_PATH)


if __name__ == '__main__':
    main()
