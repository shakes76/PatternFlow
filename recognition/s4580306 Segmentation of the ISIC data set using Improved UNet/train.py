import tensorflow as tf
import matplotlib.pyplot as plt
from modules import *
from dataset import *

LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 16


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
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=adam_optimizer, loss=dice_loss, metrics=['accuracy'])
    # print(model.summary())
    history = model.fit(train_ds.batch(BATCH_SIZE), epochs=EPOCHS, validation_data=validation_ds.batch(BATCH_SIZE))
    model.save('/tmp/model')
    plot_history(history)

    # Evaluate performance on test, model.evaluate()
    i = 0
    loss_values = []
    coefficients = []
    under = 0
    fine = 0
    for test_image, test_mask in test_ds.batch(1):
        loss, coefficient = model.evaluate(test_image, test_mask)
        loss_values.append(loss)
        coefficients.append(coefficient)

        if coefficient < 0.8:
            under += 1
        else:
            fine += 1

        i += 1

    percentage_fine = ((fine / i) * 100)
    average_dice = sum(coefficients) / len(coefficients)
    print(">>> Evaluating Test Set \n Test dataset size: " + str(i))
    print("Amount fine: " + str(fine))
    print("Amount under 0.8: " + str(under))
    print("Average Dice Coefficient: " + str(average_dice))
    print("---- " + str(percentage_fine) + "% of Test Set has 0.8 Dice Coefficient or above ----")

    plt.hist(coefficients)
    plt.title("Dice Coefficients of Test Set for Total Epochs: " + str(EPOCHS))
    plt.ylabel('Frequency')
    plt.xlabel('Dice Coefficient')
    plt.show()


if __name__ == "__main__":
    main()
