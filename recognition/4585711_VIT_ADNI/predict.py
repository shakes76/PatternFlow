import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils import init_model, configure_gpus
from dataset import get_data_preprocessing

def plot_segments(test_case, preprocessed_test_case, batch_size, image_size, cropped_image_size):
    n_row=batch_size
    n_col=2
    plt.figure(figsize=(1.0 * n_col, 1.0 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row):
        plt.subplot(n_row, n_col, i*2 + 1)
        plt.imshow(tf.reshape(test_case[i], image_size), cmap=plt.cm.gray)
        plt.title("Original", size=12)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(n_row, n_col, i*2 + 2)
        plt.imshow(tf.reshape(preprocessed_test_case[i], cropped_image_size), cmap=plt.cm.gray)
        plt.title("Preprocessed", size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig("examples.png")

if __name__ == "__main__":
    train_ds, test_ds, valid_ds, preprocessing, model, p = init_model()

    model.load_weights(p.data_dir() + "checkpoints/my_checkpoint")

    test_case = iter(test_ds).next()[0] * 255
    prediction = model.predict(test_case)
    prediction = np.argmax(prediction, axis=-1)

    plot_segments(test_case, preprocessing(test_case), p.batch_size(), p.image_size(), p.cropped_image_size())