import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def load_data():
    """
    Creates lists of the directory links for all of the images to be used in the model

    Params: None

    Returns: Three list objects containing sets of data for training, validation and testing.
    """
    # Initialise three empty lists for our data to be stored appropriately
    training_data = []
    validation_data = []
    testing_data = []

    # Create tuple pairs of the directories for the files with the images, and
    # which list they should be sorted into
    location_and_data_category = [("D:/keras_png_slices_data/keras_png_slices_train", \
            training_data), ("D:/keras_png_slices_data/keras_png_slices_vaildate", \
            validation_data), ("D:/keras_png_slices_data/keras_png_slices_test", \
            testing_data)]

    # Find and store the individual directories for each image in each file
    for data_set in location_and_data_category:
        for file_name in os.listdir(data_set[0]):
            data_set[1].append(os.path.join(data_set[0], file_name))

    return training_data, validation_data, testing_data


# for file_path in os.listdir("D:/dog"):
#     image_path = os.path.join("D:/dog", file_path)
# img = Image.open(image_path)

# img = Image.open("D:/dog/dog.png")
# plt.imshow(img)
# plt.show(block=True)

