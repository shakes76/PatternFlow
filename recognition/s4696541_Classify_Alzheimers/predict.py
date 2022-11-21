"""
Example usage of trained alzheimers classification model.
"""
import tensorflow as tf
from dataset import *
from train import MODEL_PATH

assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

if __name__ == "__main__":
    #Load model
    az_model = keras.models.load_model(MODEL_PATH)

    print("Using model:\n", az_model.summary())

    #Test model on testing dataset
    testing_ds = testing_dataset()

    loss, acc = az_model.evaluate(testing_ds)

    print(f"Model accuracy: {acc*100}, Model loss: {loss}")