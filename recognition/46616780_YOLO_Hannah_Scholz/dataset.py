# File containing the data loader for loading and preprocessing your data
import os
from os import listdir

import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import cv2

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

        # Create figure and axes
        plt.imshow(img)
        ax = plt.gca()

        bounding_box_info = generate_bounding_box(img)

        print(bounding_box_info)

        # Bounding box width:
        bounding_width = bounding_box_info[2] * 640
        print(bounding_width)

        bounding_height = bounding_box_info[3] * 640
        print(bounding_height)

        x = (bounding_box_info[0]*640)/2
        y = (bounding_box_info[1]*640)/2
        print(x)
        print(y)

        x_min = bounding_box_info[4]
        print(x_min)
        x_max = bounding_box_info[5]
        print(x_max)

        y_min = bounding_box_info[6]
        print(y_min)

        y_max = bounding_box_info[7]
        print(y_max)

        # Create a Rectangle patch
        rect = Rectangle((x_min, y_min), bounding_width, bounding_height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()

        return


def generate_bounding_box(image):
    image_array = np.array(image)

    # Compute the xMin, xMax, yMin, yMax
    # Average xMin and xMax  and yMin and yMax to compute the centre  of the bounding box
    # Divide the x average by the width of the image and the y average by the height of the image
    height, width = image.size

    # Image colours are 0 = black, 255 = white
    # Lesion is in white
    pix = image.load()

    x_min = 640
    y_min = 640

    x_max = 0
    y_max = 0

    # Iterate through each pixel the number of pixels is size of image
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            # Determine if each pixel is white or black
            if pix[x, y] == 255:
                # Colour is white
                if x < x_min:
                    x_min = x
                elif x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                elif y > y_max:
                    y_max = y

    # min function min()
    print("X min: ", x_min)
    print("X max: ", x_max)

    print("Y min: ", y_min)
    print("Y max: ", y_max)

    # Normalise
    x_avg_normalised = ((x_min + x_max) / 2) / IMAGE_SIZE
    print("x avg norm: ", x_avg_normalised)

    y_avg_normalised = ((y_min + y_max) / 2) / IMAGE_SIZE
    print("y avg norm: ", y_avg_normalised)

    # Bounding Box dimensions
    width_box = (x_max - x_min) / width
    height_box = (y_max - y_min) / height
    print("box width: ", width_box)
    print("box height: ", height_box)

    return x_avg_normalised, y_avg_normalised, width_box, height_box, x_min, x_max, y_min, y_max


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
