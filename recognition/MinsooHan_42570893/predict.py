import os
import utils
import SimpleITK as itk
import numpy as np
import pandas as pd
from skimage import io
import tensorflow
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Path of test, test GT, and test metadata
save_test_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_data/'
save_test_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_ground_truth_data/'
metadata_test = pd.read_csv('E:/Uni/COMP3710/ISIC-2017_Test_v2_Data_metadata.csv')


# Define a function that generates test data
def test_data():
    x, y = [], []
    for index, cell in metadata_test.iterrows():
        read_image = itk.ReadImage(save_test_data + cell[0] + '.png')
        image_array = itk.GetArrayFromImage(read_image) / 255.0
        mask = io.imread(save_test_ground_truth_data + cell[0] + '_segmentation.png') / 255.0
        x.append(image_array)
        y.append(mask)
    return x, y


# Define a function that loads an image and the corresponding GT image at index i.
def visualize_image(i):
    image_list = os.listdir(save_test_data)
    index = i
    image_GT_list = os.listdir(save_test_ground_truth_data)
    image = itk.ReadImage(os.path.join(save_test_data, image_list[index]))
    image_array = itk.GetArrayFromImage(image) / 255.0

    image_GT = io.imread(os.path.join(save_test_ground_truth_data, image_GT_list[index])) / 255.0

    return image_array, image_GT


# Generate test_x and test_y
test_x, test_y = test_data()
test_x = np.array(test_x)
test_y = np.array(test_y)
test_y = np.expand_dims(test_y, axis=-1)
z3 = np.zeros(test_y.shape[:-1] + (2,), dtype=test_y.dtype)
test_y = np.concatenate((test_y, z3), axis=-1)

# Load the model 'improvedUnet'
model = tensorflow.keras.models.load_model(
    'improvedUnet.h5',
    custom_objects={
        "dice_coef_loss": utils.dice_coef_loss,
        "dice_coef": utils.dice_coef,
        "LeakyReLU": tensorflow.keras.layers.LeakyReLU,
    },
)

# Visualize original test image, test GT, and a predicted GT image.
figure = plt.figure()
figure.subplots_adjust(hspace=0.4, wspace=0.4)

# Run several times changing the parameter of visualize_image()
image_array, image_GT = visualize_image(100)
test_image = np.expand_dims(image_array, axis=0)
result = model.predict(test_image)
result = result > 0.5
result = np.squeeze(result, axis=0)

show_image = figure.add_subplot(1, 3, 1)
show_image.imshow(image_array)
plt.title("Test image")

show_image = figure.add_subplot(1, 3, 2)
show_image.imshow(image_GT, cmap="gray")
plt.title("Test GT image")

show_image = figure.add_subplot(1, 3, 3)
show_image.imshow(result * 255, cmap="gray")
plt.title("Predicted GT")
plt.show()

# Store the scores.
loss, dice_coef = model.evaluate(test_x, test_y)
