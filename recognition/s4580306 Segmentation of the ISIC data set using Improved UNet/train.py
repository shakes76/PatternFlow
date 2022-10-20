import tensorflow as tf
import matplotlib.pyplot as plt
from modules import *
from dataset import *

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 30


def plot_history(history):
    """
    Plots the value of the Dice Coefficient and Accuracy throughout training and validation.
    Args:
        history: loss and coefficient history of the model during training.

    """
    plt.figure()
    plt.plot(history.history['dice_coef'], 'r', label='Training Dice')
    plt.plot(history.history['val_dice_coef'], 'bo', label='Validation Dice')
    plt.plot(history.history['accuracy'], 'gold', label='Training Loss')
    plt.plot(history.history['val_accuracy'], 'green', label='Validation Dice')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.savefig("results.png")
    plt.show()


def dice_coefficient(y_true, y_predicted):
    """
    Calculates the Dice Coefficient used in the model.
    DSC Tensorflow implementation sourced from Medium:
        [https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c]
        [16/10/2022]

    Args:
        y_true: true output
        y_predicted: output predicted by the model

    Returns: the dice coefficient for the prediction

    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_predicted_f = tf.keras.backend.flatten(y_predicted)

    intersection = tf.keras.backend.sum(y_true_f * y_predicted_f)

    dice_coeff = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_predicted_f))

    return dice_coeff


def dice_loss(y_true, y_pred):
    """
    Calculates the Dice Coefficient Loss function
    Args:
        y_true: true output
        y_pred: output predicted by the model

    Returns: dice loss

    """
    return 1 - dice_coefficient(y_true, y_pred)


def main():
    train_ds, test_ds, validation_ds = data_loader()
    improved_unet = ImprovedUNET()
    model = improved_unet.data_pipe_line()
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=adam_optimizer, loss=dice_loss, metrics=['accuracy', dice_coefficient])
    print(model.summary())
    history = model.fit(train_ds.batch(BATCH_SIZE), epochs=EPOCHS, validation_data=validation_ds.batch(BATCH_SIZE))

    plot_history(history)

    # Evaluate performance on test, model.evaluate()
    i = 0
    lossV = []
    coefficientV = []
    under = 0
    fine = 0
    for test_image, test_mask in test_ds.batch(1):
        loss, coefficient = model.evaluate(test_image, test_mask)
        lossV.append(loss)
        coefficientV.append(coefficient)

        if (coefficient < 0.8):
            under += 1
        else:
            fine += 1

        i += 1

    percentageFine = ((fine / i) * 100);
    averageDC = sum(coefficientV) / len(coefficientV)
    print(">>> Evaluating Test Set \n Test dataset size: " + str(i))
    print("Amount fine: " + str(fine))
    print("Amount under 0.8: " + str(under))
    print("Average Dice Coefficient: " + str(averageDC))
    print("---- " + str(percentageFine) + "% of Test Set has 0.8 Dice Coefficient or above ----")

    plt.hist(coefficientV)
    plt.title("Dice Coefficients of Test Set for Total Epochs: " + str(EPOCHS))
    plt.ylabel('Frequency')
    plt.xlabel('Dice Coefficient')
    plt.show()



if __name__ == "__main__":
    main()