"""
Example usage of trained alzheimers classification model.
"""
import tensorflow as tf
from modules import AlzheimerModel
from dataset import *

assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

if __name__ == "__main__":
    #az_model = keras.models.load_model("az_model")

    az_model = AlzheimerModel(
        num_patches=NUM_PATCHES, 
        num_layers=6,
        num_heads=12,
        d_model=768,
        d_mlp=3072,
        head_layers=1000,
        dropout_rate=0.2,
        num_classes=2
    )

    az_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    az_model.build([1, 240, 240, 3])

    print("Using model:\n", az_model.summary())

    testing_ds = testing_dataset()

    loss, acc = az_model.evaluate(testing_ds)

    print(f"Model accuracy: {acc*100}, Model loss: {loss}")