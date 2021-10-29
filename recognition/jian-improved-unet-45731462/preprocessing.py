"""
[INSERT]

@author Jian Yang Lee
@email jianyang.lee@uqconnect.edu.au
"""

import numpy as np
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from improved_model import model
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical


def to_numpy(path, image_type):
    """
    [INSERT]
    """
    numpy_images = []

    for filename in os.listdir(path):
        if filename != ".DS_Store":
            if image_type == "bw":
                read_img = cv2.imread(os.path.join(path, filename), 0)
            elif image_type == "rgb":
                read_img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
            new_img = cv2.resize(read_img, (128, 96))
            numpy_images.append(new_img)

    return np.array(numpy_images)

def encode_bw(input):
    """[summary]

    Args:
        input ([type]): [description]

    Returns:
        [type]: [description]
    """

    dim, height, width = input.shape

    # converts to 1D
    input = np.ravel(input)

    # ensures the input image is either values of 0 or 1
    input[(input >= 0) & (input < 128)] = 0
    input[(input >= 128) & (input < 256)] = 1

    return input.reshape(dim, height, width)


def main():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # read images from file directory, and ensure variables with 4 dimensions
    inputs = to_numpy("jian-improved-unet-45731462/Input_x2", "rgb")
    labels = to_numpy("jian-improved-unet-45731462/GroundTruth_x2", "bw")

    # converts label mask to either 0 or 1 value, expand to 4 dimensions
    labels = encode_bw(labels)
    labels = np.expand_dims(labels, axis=3)
    labels = to_categorical(labels, num_classes=2)

    # print(inputs.shape)
    # print(labels.shape)

    # train test split
    train = 0.7
    valid= 0.15
    test = 0.15

    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=1-train)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=test/(test + valid))

    height = 96
    width = 128
    input_channel = 3
    desired_channel = 2

    ready_model = model(height, width, input_channel, desired_channel)
    ready_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # model summary
    print(ready_model.summary())


    # # train model
    history = ready_model.fit(x_train, y_train, 
               batch_size=2, verbose=1, 
               epochs=15, 
               validation_data=(x_valid, y_valid), 
               shuffle=False)

    # # save model
    ready_model.save("Improved_Jian_Epoch15_Batch2.hdf5")


if __name__ == "__main__":
    main()
