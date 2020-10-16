from recognition.laterality_classification.laterality_classifier import *
import glob
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load_images(dir_data, Ntrain, Ntest):
    nm_imgs = np.sort(os.listdir(dir_data))
    ## name of the jpg files for training set
    nm_imgs_train = nm_imgs[:Ntrain]
    ## name of the jpg files for the testing data
    nm_imgs_test = nm_imgs[Ntrain:Ntrain + Ntest]
    img_shape = (64, 52, 3)

    def get_data(nm_imgs_train):
        X_train = []
        for i, myid in enumerate(nm_imgs_train):
            image = load_img(dir_data + "/" + myid,
                             target_size=img_shape[:2])
            image = img_to_array(image) / 127.0

            X_train.append(image)
        X_train = np.array(X_train)
        return (tf.convert_to_tensor(X_train))


if __name__ == "__main__":
    data_dir = "H:/Desktop/AKOA_Analysis/"
    load_images(data_dir)
    classifier = LateralityClassifier
