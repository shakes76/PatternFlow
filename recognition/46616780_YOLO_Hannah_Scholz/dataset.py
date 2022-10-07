# File containing the data loader for loading and preprocessing your data
import os
from os import listdir
from PIL import Image

IMAGE_SIZE = 256


def load_images(image_path):
    file_names = listdir(image_path)

    for new_filename in file_names:
        if new_filename.endswith(".DS_Store"):
            continue

        with Image.open(image_path + new_filename) as img:
            width, height = img.size

            # Change the shape of the images so all the images are the same size
            # And all the mask sizes too 256x256
            if width > height:
                img = img.resize((IMAGE_SIZE, round(height * (IMAGE_SIZE / width))))
            else:
                img = img.resize((round(height * (IMAGE_SIZE / width)), IMAGE_SIZE))


            bounding_box(img)



def bounding_box(image):
    print(image.size)


training_data_path = "Datasets/Training/training_data/"
for filename in os.listdir(training_data_path):
    if filename.endswith(".png"):
        os.remove(training_data_path + filename)
load_images(training_data_path)


testing_data_path = "Datasets/Testing/testing_data/"
for filename in os.listdir(testing_data_path):
    if filename.endswith(".png"):
        os.remove(testing_data_path + filename)
load_images(testing_data_path)
