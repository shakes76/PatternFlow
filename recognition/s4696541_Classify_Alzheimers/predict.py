"""
Example usage of trained alzheimers classification model.
"""
import tensorflow as tf
from modules import AlzheimerModel
from dataset import *

assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

if __name__ == "__main__":
    az_model = keras.models.load_model("az_model")

    print("Using model:\n", az_model.summary())

    testing_ds = testing_dataset()

    loss, acc = az_model.evaluate(testing_ds)

    print(f"Model accuracy: {acc*100}, Model loss: {loss}")