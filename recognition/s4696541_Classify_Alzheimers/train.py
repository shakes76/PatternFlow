"""
Source code for training, validating, testing, and saving alzheimers classification model.
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import optimizers, losses
from dataset import *
import matplotlib.pyplot as plt

from modules import AlzheimerModel

tf.get_logger().setLevel('INFO')
assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

EPOCHS = 25

if __name__ == "__main__":
    az_model = AlzheimerModel(
        num_patches=NUM_PATCHES, 
        num_layers=2,
        num_heads=2,
        d_model=D_MODEL,
        d_mlp=2000,
        head_layers=100,
        dropout_rate=0.2,
        num_classes=2
    )

    az_model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=tfa.optimizers.AdamW(0.1, 0.0005),
        metrics=['accuracy']
    )

    az_model.build((1, IMAGE_DIM, IMAGE_DIM, 3))
    print(az_model.summary())

    train_ds = training_dataset()
    validation_ds = validation_dataset()

    history = az_model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=validation_ds)

    az_model.save("az_model")

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(1, EPOCHS+1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()
    plt.savefig("model_accuracy.png")

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    plt.savefig("model_loss.png")