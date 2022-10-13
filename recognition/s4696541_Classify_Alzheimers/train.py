"""
Source code for training, validating, testing, and saving alzheimers classification model.
"""
import tensorflow as tf
import tensorflow.keras as keras
from dataset import *
import matplotlib.pyplot as plt

from modules import AlzheimerModel

tf.get_logger().setLevel('INFO')
assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"

EPOCHS = 20

if __name__ == "__main__":
    az_model = AlzheimerModel(
        num_patches=NUM_PATCHES, 
        num_layers=12,
        num_heads=12,
        d_model=768,
        d_mlp=3072,
        head_layers=3072,
        dropout_rate=0.2,
        num_classes=2
    )

    az_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    train_ds = training_dataset()
    test_ds = testing_dataset()

    history = az_model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=test_ds)

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