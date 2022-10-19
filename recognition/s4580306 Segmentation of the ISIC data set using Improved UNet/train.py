import tensorflow as tf
import matplotlib.pyplot as plt
from modules import *
from dataset import *

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 30


def plot_history(history):
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


def generate_prediction(test_data, model):
    prediction_batch = 20
    batched_test_data = test_data.batch(prediction_batch)
    test_image, test_mask = next(iter(batched_test_data))
    mask_prediction = model.predict(test_image)

    # Plot the original image, ground truth and result from the network.
    for i in range(prediction_batch):
        plt.figure(figsize=(10, 10))

        # Plot the test image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image[i])
        plt.title("Input Image")
        plt.axis("off")

        # Plot the test mask
        plt.subplot(1, 3, 2)
        plt.imshow(test_mask[i], cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        # Plot the resultant mask
        plt.subplot(1, 3, 3)

        # Display 0 or 1 for classes
        prediction = tf.where(mask_prediction[i] > 0.5, 1.0, 0.0)
        plt.imshow(prediction)
        plt.title("Resultant Mask")
        plt.axis("off")

        plt.show()
    return


def dice_coefficient(y_true, y_predicted):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_predicted_f = tf.keras.backend.flatten(y_predicted)

    intersection = tf.keras.backend.sum(y_true_f * y_predicted_f)

    dice_coefficient = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_predicted_f))

    return dice_coefficient


def dice_loss(y_true, y_pred):
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

    generate_prediction(test_ds, model)


if __name__ == "__main__":
    main()