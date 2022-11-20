# File containing the data loader for loading and preprocessing your data

import os
from os import listdir
from PIL import Image as im
import csv

# Constants in the code
IMAGE_SIZE = 640

VALIDATION_DATA_PATH = "Datasets/Validation/validation_data/"
TESTING_DATA_PATH = "Datasets/Testing/testing_data/"
TRAINING_DATA_PATH = "Datasets/Training/training_data/"

VALIDATION_MASK_PATH = "Datasets/Validation/validation_mask/"
TESTING_MASK_PATH = "Datasets/Testing/testing_mask/"
TRAINING_MASK_PATH = "Datasets/Training/training_mask/"

VALIDATION_MASK_CSV_PATH = "Datasets/Validation/validation_mask.csv"
TESTING_MASK_CSV_PATH = "Datasets/Testing/testing_mask.csv"
TRAINING_MASK_CSV_PATH = "Datasets/Training/training_mask.csv"


# Function used to resize the provided images into a suitable size for YoloV5
def load_resize_images(image_path):
    file_names = listdir(image_path)

    for filename in file_names:
        if filename.endswith(".DS_Store"):
            continue

        with im.open(image_path + filename) as img:
            # Change the shape of the images so all the images have dimensions of 640x640
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            # Save image
            img.save(image_path + filename)


# Function that determines the bounding box coordinates from the given image and csv class files.
def save_bounding_box_images(image_path, csv_path):
    file_names = listdir(image_path)

    for filename in file_names:
        # Mac files to ignore
        if filename.endswith(".DS_Store"):
            continue

        with im.open(image_path + filename) as img:
            # Call the function that given an image determines the coordinates of the bounding box
            bounding_box_info = generate_bounding_box(img)

            # Open the csv file
            name = filename.replace('_segmentation.png', '')
            file = open(csv_path)
            csvreader = csv.reader(file)

            # One hot encoding for melanoma or not using the first column of the .csv file
            for row in csvreader:
                # Check that the name of the image is same and name the text file by this name
                if row[0] == name:
                    # Get the second column
                    fileNew = open(name + '.txt', 'w')
                    fileNew.write(
                        f'{row[1]} {bounding_box_info[0]} {bounding_box_info[1]} {bounding_box_info[2]} {bounding_box_info[3]}')


# Function that determines the bounding box of a given image (only used for the mask images in black and white)
def generate_bounding_box(image):
    height, width = image.size

    # Image colours are 0 = black, 255 = white
    # Lesion is in white
    pix = image.load()

    x_min = 640
    y_min = 640
    x_max = 0
    y_max = 0

    # Iterate through each pixel, the number of pixels is the size of image
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            # Determine if each pixel is white and whether it is a max or min
            if pix[x, y] == 255:
                if x < x_min:
                    x_min = x
                elif x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                elif y > y_max:
                    y_max = y

    # Normalise
    x_avg_normalised = ((x_min + x_max) / 2) / IMAGE_SIZE
    y_avg_normalised = ((y_min + y_max) / 2) / IMAGE_SIZE

    # Bounding Box dimensions
    width_box = (x_max - x_min) / width
    height_box = (y_max - y_min) / height

    return x_avg_normalised, y_avg_normalised, width_box, height_box, x_min, x_max, y_min, y_max


# Function to clean up the images in the directory - removes all images with .png that were unnecessary for the data
def clean_up_directory(image_directory):
    print(image_directory)
    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            os.remove(image_directory + filename)


def main():
    # Do this for all three directories
    # Not necessary for mask data
    clean_up_directory(VALIDATION_DATA_PATH)
    clean_up_directory(TESTING_DATA_PATH)
    clean_up_directory(TRAINING_DATA_PATH)

    # Resize all images
    load_resize_images(VALIDATION_DATA_PATH)
    load_resize_images(TESTING_DATA_PATH)
    load_resize_images(TRAINING_DATA_PATH)

    load_resize_images(VALIDATION_MASK_PATH)
    load_resize_images(TESTING_MASK_PATH)
    load_resize_images(TRAINING_MASK_PATH)

    # Determine bounding boxes for the mask data
    save_bounding_box_images(VALIDATION_MASK_PATH, VALIDATION_MASK_CSV_PATH)
    save_bounding_box_images(TESTING_MASK_PATH, TESTING_MASK_CSV_PATH)
    save_bounding_box_images(TRAINING_MASK_PATH, TRAINING_MASK_CSV_PATH)


if __name__ == '__main__':
    main()
