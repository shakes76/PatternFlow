"""
This is the script for the prediction of the siamese model.
Through input the path of the images, the model will predict whether the two images are the same person.
"""


import dataset
import modules

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


# Import the path of the dataset that you want to classify
TEST_PATH = './AD_NC/test'
IMG1_PATH = './AD_NC/test/AD'
IMG2_PATH = './AD_NC/test/NC'

# The shape of the images 
INPUT_SHAPE = (120, 128)
COLOR_MODE = 'grayscale'

def main():
    '''
    The main function used to predict and show some results of the model.
    '''
    #######################################################
    # A visuliazer to show the similarity of the two images
    #######################################################

    # create the model
    siamese = modules.SiameseModel()

    # compile the model
    siamese.compile(optimizer=tf.keras.optimizers.Adam(0.00006))

    # load the weights of the model
    siamese.load_weights('training4/cp-0025.ckpt')

    # load the data
    dataset1 = dataset.load_data(IMG1_PATH, INPUT_SHAPE, COLOR_MODE)
    dataset2 = dataset.load_data(IMG2_PATH, INPUT_SHAPE, COLOR_MODE)
    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = dataset.make_pair(dataset1, dataset2)
    test_dataset = dataset.shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)
    test_dataset = test_dataset.batch(16)

    # visualize the the prediction of the model on the given dataset
    for img1, img2, label in test_dataset.take(1):
        dataset.visualize(img1, img2, label, to_show=4, num_col=4, predictions=siamese([img1, img2], training=False).numpy(), test=True)
    

    #######################################################
    # A classifier used to classify the images
    #######################################################
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH, labels = 'inferred', color_mode = COLOR_MODE, class_names =['AD', 'NC'], image_size=INPUT_SHAPE, batch_size=32
    )

    classifier = modules.classifier()
    classifier.fit(test_ds, epochs=1)

    # predict some images and compare the results
    for img, label in test_ds.take(1):
        predictions = classifier.predict(img)
        # compare the predictions with the labels
        print(np.argmax(predictions, axis=1))
        print(label.numpy())


if __name__ == '__main__':
    main()