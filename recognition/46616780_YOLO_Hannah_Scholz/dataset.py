# File containing the data loader for loading and preprocessing your data
import os
from os import listdir
from PIL import Image

IMAGE_SIZE = 640
VALIDATION_DATA_PATH = "Datasets/Validation/validation_data"
TESTING_DATA_PATH = "Datasets/Testing/testing_data/"
TRAINING_DATA_PATH = "Datasets/Training/training_data/"


def load_images(image_path):
    file_names = listdir(image_path)

    for new_filename in file_names:
        if new_filename.endswith(".DS_Store"):
            continue

        with Image.open(image_path + new_filename) as img:
            width, height = img.size

            # Change the shape of the images so all the images are the same size
            # And all the mask sizes = 640x640
            if width > height:
                img = img.resize((IMAGE_SIZE, round(height * (IMAGE_SIZE / width))))
            else:
                img = img.resize((round(width * (IMAGE_SIZE / height)), IMAGE_SIZE))

            generate_bounding_box(img)


def generate_bounding_box(image):
    print(image.size)


def clean_up_directory(image_directory):
    print(image_directory)
    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            os.remove(image_directory + filename)
    # load_images(image_directory)


def main():
    # Do this for all three directories
    clean_up_directory(VALIDATION_DATA_PATH)
    clean_up_directory(TESTING_DATA_PATH)
    clean_up_directory(TRAINING_DATA_PATH)


if __name__ == '__main__':
    main()
