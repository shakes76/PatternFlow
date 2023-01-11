from dataset import *
from modules import *
from train import *
import tensorflow as tf
import matplotlib.pyplot as plt

def calculate_ssim(original_data, predicted_data):
    """ Calculates and the average of the SSIM of all images in the two sets of data as a percentage. """
    ssim = tf.image.ssim(original_data, predicted_data, max_val=1)
    print("SSIM of data sets:", ssim)

def compare_predicted(original_data, predicted_data, index):
    """ Plots the original and predicted image at the given index """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(original_data[index])
    ax.set_title("Original Image")

    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(predicted_data[index])
    ax.set_title("Reconstructed Image")
    fig.show()

def plot_loss(model):
    return None

training_data, validation_data, testing_data, data_variance = load_data()
model = train_vqvae()

predictions = model.predict(testing_data)

calculate_ssim(testing_data, predictions)
compare_predicted(testing_data, predictions, 8)