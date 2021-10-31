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

    Source: https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

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
        validation_split = 0.1,
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

if __name__ == "__main__":
    batch_size = 16
    depth = 16
    epochs = 100
    n = 192
    m = 256

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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=dice_coefficient_loss, metrics=['accuracy', dice_coefficient])
    output = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))
    
    print(output.history['loss'])
    print(output.history['accuracy'])
    print(output.history['dice_coefficient'])

    model.save("saved_model")
    
    plot_prediction(model, X_test, y_test)
