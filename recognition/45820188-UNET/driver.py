"""
Driver script to run the ISICs 2018 Dataset through
the Improved UNET Model.

@author Andrew Luong (45820188)

Created: 29/10/2021
"""

from model import build_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice Coefficient required model compiling metrics

    Source: https://www.jeremyjordan.me/semantic-segmentation/
    """
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * tf.reduce_mean(y_pred * y_true, axes)
    denominator = tf.reduce_mean(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    
    return 1 - tf.reduce_mean((numerator + smooth) / (denominator + smooth))

def dice_coefficient_loss(y_true, y_pred):
    """
    Calculates the dice coefficient loss. 
    """
    return 1 - dice_coefficient(y_true, y_pred)

def process_images(path, segmentation):
    """
    Uses Keras function to convert a directory of images 
    into a Keras Dataset

    Shuffles the images with a seed of my student ID
    """
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory = path, 
        labels=None,
        label_mode = 'binary',
        batch_size = batch_size,
        validation_split = 0.2,
        subset=segmentation,
        image_size = (n, m),
        color_mode = 'grayscale',
        shuffle = True,
        seed = 45820188
    )

def plot_prediction(model, X_test, y_test):
    """
    Takes the model and predicts the output from the given image set

    Plots the images with Original vs Prediction vs Expected
    """
    prediction = model.predict(X_test)
    plt.figure(figsize=(10, 10))
    n = 4
    for i in range(n):
        plt.subplot(n, 3, i*3+1)
        plt.imshow(X_test[i])
        plt.axis('off')
        plt.title("Original", size=11)

        plt.subplot(n, 3, i*3+2)
        
        plt.imshow(tf.cast(prediction[i] * 255.,'uint8'))
        plt.axis('off')
        plt.title("Prediction", size=11)

        plt.subplot(n, 3, i*3+3)
        plt.imshow(y_test[i])
        plt.axis('off')
        plt.title("Expected", size=11)
    plt.show()

def plot_dice_coefficient(output):
    """
    Plots the dice coefficient value over the epochs

    Compares the Train and Validate sets
    """
    dice = output.history['dice_coefficient']
    val_dice = output.history['val_dice_coefficient']

    plt.plot(output.epoch, dice, 'b', label='Train')
    plt.plot(output.epoch, val_dice, 'r', label='Validate')

    plt.ylim([0, 1])
    
    plt.title('Dice Coefficient Value over Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Dice Coefficient')
    plt.legend(loc="lower right")
    plt.show()

def plot_loss(output):
    """
    Plots the loss value over the epochs

    Compares the Train and Validate Sets
    """
    loss = output.history['loss']
    val_loss = output.history['val_loss']

    plt.plot(output.epoch, loss, 'b', label='Training Loss')
    plt.plot(output.epoch, val_loss, 'r', label='Validation Loss')

    plt.ylim([0, 1])

    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(loc="upper right")
    plt.show()



if __name__ == "__main__":
    batch_size = 16
    depth = 16
    epochs = 1000
    n = 96
    m = 128
    
    # Load the Improved UNET Model
    model = build_model(input_shape=(n, m, 1), depth=depth)
    model.summary()
    
    # Full Dataset
    X_train_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1-2_Training_Input_x2", "training")
    y_train_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1_Training_GroundTruth_x2", "training")

    X_test_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1-2_Training_Input_x2", "validation")
    y_test_ds = process_images("C:\ISIC Dataset\Full Set\ISIC2018_Task1_Training_GroundTruth_x2", "validation")

    # Converts the tf dataset into array of images that can be used by the model
    X_train = tf.concat([x for x in X_train_ds], axis=0)
    y_train = tf.concat([x for x in y_train_ds], axis=0)
    X_test = tf.concat([x for x in X_test_ds], axis=0)
    y_test = tf.concat([x for x in y_test_ds], axis=0)

    X_train = X_train / 255.
    y_train = y_train / 255.
    X_test = X_test / 255.
    y_test = y_test / 255.

    model.compile(optimizer='adam', loss=dice_coefficient_loss, metrics=[dice_coefficient, 'accuracy'])
    output = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))
    
    model.save("saved_model")

    total = sum(output.history['dice_coefficient'])
    print("Average Dice: " + str(total/epochs))
    
    plot_prediction(model, X_test, y_test)
    plot_dice_coefficient(output)
    plot_loss(output)
