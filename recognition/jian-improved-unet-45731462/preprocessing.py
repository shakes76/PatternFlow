"""
[INSERT]

@author Jian Yang Lee
@email jianyang.lee@uqconnect.edu.au
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from improved_model import model
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt


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

def dice(predict, test, smooth=1):
    """[summary]

    Args:
        predict ([type]): [description]
        test ([type]): [description]
        smooth (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    intersection = np.sum(predict.flatten() * test.flatten())
    return (2. * intersection + smooth) / (np.sum(predict) + np.sum(test) + smooth)

def loss_graph(results, epoch_num):
    """[summary]

    Args:
        results ([type]): [description]
        epoch_num ([type]): [description]
    """

    loss = results.history["loss"]
    val_loss = results.history["val_loss"]
    plt.plot(range(1, epoch_num+1), loss, 'k', label='Training')
    plt.plot(range(1, epoch_num+1), val_loss, 'r', label='Validation')
    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def accuracy_graph(results, epoch_num):
    """[summary]

    Args:
        results ([type]): [description]
        epoch_num ([type]): [description]
    """
    loss = results.history["accuracy"]
    val_loss = results.history["val_accuracy"]
    plt.plot(range(1, epoch_num+1), loss, 'k', label='Training')
    plt.plot(range(1, epoch_num+1), val_loss, 'r', label='Validation')
    plt.title('Accuracy Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # read images from file directory, and ensure variables with 4 dimensions
    inputs = to_numpy("Input_x2", "rgb")
    labels = to_numpy("GroundTruth_x2", "bw")

    # converts label mask to either 0 or 1 value, expand to 4 dimensions
    labels = encode_bw(labels)
    labels = np.expand_dims(labels, axis=3)
    labels = to_categorical(labels, num_classes=2)

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
    epoch = 30

    ready_model = model(height, width, input_channel, desired_channel)
    ready_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # model summary
    # print(ready_model.summary())

    ## train model
    # results = ready_model.fit(x_train, y_train, 
    #            batch_size=2, verbose=1, 
    #            epochs=epoch, 
    #            validation_data=(x_valid, y_valid), 
    #            shuffle=False)

    ## save model
    # ready_model.save("Improved_Jian_Epoch30_Batch2.hdf5")

    # load weights from previous trained models
    ready_model.load_weights("Improved_Jian_Epoch30_Batch2.hdf5")

    # plotting losses and accuracy
    # loss_graph(results, epoch)
    # accuracy_graph(results, epoch)

    # evaluate model and get accuracy
    _, accuracy = ready_model.evaluate(x_test, y_test)
    print(f"Accuracy -> {accuracy * 100}%")

    predicted = ready_model.predict(x_test)

    # dice scores for each 2 channels
    for c in range(desired_channel):
        coeff = dice(predicted[:,:,:,c], y_test[:,:,:,c])
        print(f"Dice Score for Channel {c}: {coeff}")


if __name__ == "__main__":
    main()
