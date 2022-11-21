"""
This is the script to train the Siamese Network.
"""

import dataset
import modules

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# set the path to the training data and the test data
TRAIN_AD_PATH = './AD_NC/train/AD'
TRAIN_NC_PATH = './AD_NC/train/NC'
TEST_AD_PATH = './AD_NC/test/AD'
TEST_NC_PATH = './AD_NC/test/NC'

# set the parameters of the input images
INPUT_SHAPE = (120, 128)
COLOR_MODE = 'grayscale'

# set the batch size and the training size
BATCH_SIZE = 16
TRAINING_SIZE = 1800

# set the mode of the program
TRAINING_MODE = True
VISUALIZE = True


def main():
    '''
    The main function used to training, validating and testing the model.
    and visualize the result of the model.
    '''
    
    # load the Train and Validation data
    AD_dataset = dataset.load_data(TRAIN_AD_PATH, INPUT_SHAPE, COLOR_MODE)
    NC_dataset = dataset.load_data(TRAIN_NC_PATH, INPUT_SHAPE, COLOR_MODE)
    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = dataset.make_pair(AD_dataset, NC_dataset)
    choice_dataset = dataset.shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)
    train_dataset, validation_dataset = dataset.split_dataset(choice_dataset, BATCH_SIZE, TRAINING_SIZE)

    # load the Test data
    AD_test_dataset = dataset.load_data(TEST_AD_PATH, INPUT_SHAPE, COLOR_MODE)
    NC_test_dataset = dataset.load_data(TEST_NC_PATH, INPUT_SHAPE, COLOR_MODE)
    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = dataset.make_pair_test(AD_dataset, NC_dataset, AD_test_dataset, NC_test_dataset)
    test_dataset = dataset.shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)
    test_dataset = test_dataset.batch(16)

    # Visualize the training data
    if VISUALIZE:
        for img1, img2, labels in train_dataset.take(1):
            dataset.visualize(img1, img2, labels)


    # set up the callbacks to save the weights 
    checkpoint_path = "training4/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')

    # record the training history
    csv_logger = tf.keras.callbacks.CSVLogger('training3.log',append=False)

    # create the model
    siamese = modules.SiameseModel()

    # compile the model 
    siamese.compile(optimizer=tf.keras.optimizers.Adam(0.00006))

    # Start the training if TRAINING_MODE is True
    if TRAINING_MODE:

        # train the model
        history = siamese.fit(train_dataset, epochs=25, validation_data=validation_dataset, callbacks=[cp_callback, csv_logger])

        if VISUALIZE:
            # plot the training loss and accuracy
            N = 10
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy on Siamese Network")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig("plot.png")

    else:
        # load the weights of the model
        siamese.load_weights('training4/cp-0025.ckpt')

    # evaluate the model on the validation data
    loss, accuracy = siamese.evaluate(validation_dataset)

    # visualize the the prediction of the model on the validation data
    if VISUALIZE:
        for img1, img2, label in validation_dataset.take(1):
            dataset.visualize(img1, img2, label, to_show=4, num_col=4, predictions=siamese([img1, img2], training=False).numpy(), test=True)

    # evaluate the model on the test data
    loss, accuracy = siamese.evaluate(test_dataset)

    # visualize the the prediction of the model on the test data
    if VISUALIZE:
        for img1, img2, label in test_dataset.take(1):
            dataset.visualize(img1, img2, label, to_show=4, num_col=4, predictions=siamese([img1, img2], training=False).numpy(), test=True)


if __name__ == '__main__':
    main()