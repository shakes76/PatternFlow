import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Cropping2D, RandomTranslation, RandomZoom
from keras import Sequential

from utils import init_model
from dataset import TrackCrop

def plot_preprocessing(test_case, preprocessed_test_case, image_size, preprocessed_image_size, image_dir, name):
    test_case = test_case[0]
    preprocessed_test_case = preprocessed_test_case[0]

    n_row = 1
    n_col = 2
    plt.figure(figsize=(1.6 * n_col, 2.0 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    plt.subplot(n_row, n_col, 1)
    plt.imshow(tf.reshape(test_case, image_size), cmap=plt.cm.gray)
    plt.title("Original", size=12)
    plt.xticks(())
    plt.yticks(())

    plt.subplot(n_row, n_col, 2)
    plt.imshow(tf.reshape(preprocessed_test_case, preprocessed_image_size), cmap=plt.cm.gray)
    plt.title("Preprocessed", size=12)
    plt.xticks(())
    plt.yticks(())

    plt.savefig(image_dir + name + ".png")

def plot_predictions(test_case, ground_truth, prediction, image_size, image_dir):
    n_row=2
    n_col=8
    plt.figure(figsize=(1.6 * n_col, 1.4 * n_row))
    plt.subplots_adjust(bottom=0.01, left=.01, right=.99, top=.88, hspace=0.8, wspace=0.0)
    for i in range(n_row):
        for j in range(n_col):
            ind = i*n_col + j 

            ground_truth_label = "Alzheimer's" if ground_truth[ind] == 0 else "Normal"
            prediction_label = "Alzheimer's" if prediction[ind] == 0 else "Normal"
            colour = "g" if ground_truth[ind] == prediction[ind] else "r"

            ax = plt.subplot(n_row, n_col, ind + 1)
            ax.figure
            ax.set_facecolor(colour)
            plt.imshow(tf.reshape(test_case[ind], image_size), cmap=plt.cm.gray)
            plt.title(f"Ground Truth: {ground_truth_label}\nPrediction: {prediction_label}", size=8)
            plt.xticks(())
            plt.yticks(())

    plt.savefig(image_dir + "predictions.png")

def generate_preprocessing_images(test_case, p):
    test_case = tf.expand_dims(test_case, 0)

    preprocessing = Cropping2D((8, 0))
    plot_preprocessing(
        test_case, preprocessing(test_case), p.image_size(), (p.image_size()[0]-16,p.image_size()[1]),
        p.image_dir(), "simple_crop")

    preprocessing = TrackCrop(p.cropped_image_size())
    plot_preprocessing(
        test_case, preprocessing(test_case), p.image_size(), p.cropped_image_size(),
        p.image_dir(), "track_crop")

    plot_preprocessing(
        test_case, test_case, p.image_size(), p.image_size(),
        p.image_dir(), "normalisation")
    
    preprocessing = Sequential([
        RandomTranslation(0.0, (-0.2,0.0), fill_mode='constant'),
        RandomZoom((-0.05, 0.1), fill_mode='constant')
    ])
    plot_preprocessing(
        test_case, preprocessing(test_case, training=True), p.image_size(), p.image_size(),
        p.image_dir(), "augmented")

def plot_examples(test_case, image_size, image_dir):
    n_row = 4
    n_col = 8

    plt.figure(figsize=(1.0 * n_col, 1.0 * n_row))
    plt.subplots_adjust(bottom=0.0, left=0.01, right=0.99, top=1.0, wspace=0.05, hspace=0.0)
    for i in range(n_row):
        for j in range(n_col):
            plt.subplot(n_row, n_col, i*n_col + j + 1)
            plt.imshow(tf.reshape(test_case[i], image_size), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())

    plt.savefig(image_dir + "examples.png")

if __name__ == "__main__":
    train_ds, test_ds, valid_ds, preprocessing, model, p = init_model()

    model.load_weights(p.data_dir() + "checkpoints/my_checkpoint")

    result = model.evaluate(test_ds)
    print("Test loss:", result[0])
    print("Test Accuracy:", result[1])

    test_case, ground_truth = iter(test_ds).next()

    generate_preprocessing_images(test_case[0], p)

    plot_examples(test_case, p.image_size(), p.image_dir())

    prediction = model(test_case).numpy()
    prediction = np.argmax(prediction, axis=-1)
    ground_truth = np.argmax(ground_truth.numpy(), axis=-1)

    plot_predictions(test_case, ground_truth, prediction, p.image_size(), p.image_dir())