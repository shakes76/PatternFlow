import tensorflow as tf
import matplotlib.pyplot as plt
from modules import *
from dataset import *


def plot_history(history):
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'])
    plt.title('Training Vs Validation Dice Loss')
    plt.legend(loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("dice_loss.png")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['accuracy'], label='Training Loss')
    plt.plot(history.history['val_accuracy'])
    plt.title('Training Vs Validation Accuracy')
    plt.legend(loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.savefig("accuracy.png")
    plt.show()


def calculate_dice_coefficient(y_true, y_predicted):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_predicted_f = tf.keras.backend.flatten(y_predicted)

    intersection = tf.keras.backend.sum(y_true_f * y_predicted_f)

    dice_coefficient = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_predicted_f))

    return dice_coefficient


def dice_loss(y_true, y_pred):
    return 1 - calculate_dice_coefficient(y_true, y_pred)


def main():
    train_ds, test_ds, validation_ds = data_loader()
    improved_unet = ImprovedUNET()
    model = improved_unet.data_pipe_line()
    model.compile(optimizer="adam", loss=dice_loss, metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_ds.batch(16), epochs=15, validation_data=validation_ds.batch(16))

    plot_history(history)


if __name__ == "__main__":
    main()
