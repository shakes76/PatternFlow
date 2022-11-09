"""
This script loads the input and output images from the ISIC dataset and performs pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
@date: 31/10/2021
@license: Attribution-NonCommercial 4.0 International. See license.txt for more details.
"""
import tensorflow as tf
from data_loader import load_data
from model import create_model 
from utilities import dice_similarity, display_images, plot_accuracy 

def main():
    """
    The main function that runs this script
    """
    # load processed data using data_loader and process_data modules
    train_data, val_data, test_data = load_data()
    # create an improved unet model
    improved_unet_model = create_model(2)
    # training parametets
    EPOCHS = 10
    BATCH_SIZE = 10
    # train model
    improved_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_similarity])
    history = improved_unet_model.fit(train_data.batch(BATCH_SIZE), epochs=EPOCHS, validation_data=val_data.batch(BATCH_SIZE))
    # plot accuracy graphs
    plot_accuracy(history)
    # evaluate on test dataset
    improved_unet_model.evaluate(test_data.batch(BATCH_SIZE))
    # display scan, mask and predicted mask from test dataset
    for scan, label in test_data.take(3):
        predicted_mask = improved_unet_model.predict(scan[tf.newaxis])
        predicted_mask = tf.argmax(predicted_mask[0], axis=-1)
        display_images([tf.squeeze(scan), tf.argmax(label, axis=-1), 
            predicted_mask], figsize=(6, 6), cmap='gray')


# run main function
if __name__ == "__main__":
    main()