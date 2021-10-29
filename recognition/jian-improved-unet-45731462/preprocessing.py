"""
Model Architecture of the Improved Unet

@author Jian Yang Lee
@email jianyang.lee@uqconnect.edu.au
"""


import numpy as np
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def to_numpy(path):
    """
    [INSERT]
    """
    numpy_images = []

    for filename in os.listdir(path):
        if filename != ".DS_Store":
            read_img = cv2.imread(os.path.join(path, filename), 0)
            new_img = cv2.resize(read_img, (128, 128))
            numpy_images.append(new_img)
    
    return np.array(numpy_images)



def preprocess_data(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):

    inputs = to_numpy("jian-improved-unet-45731462/Input_x2")
    labels = to_numpy("jian-improved-unet-45731462/GroundTruth_x2")

    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=1-train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + valid_ratio))


    # maintain aspect ratio of image when .shape


    # print(x_train.shape)
    # print(x_val.shape)
    # print(x_test.shape)


    # print(inputs.shape)
    # print(labels.shape)

    print(labels[0])

def main():
    preprocess_data()




if __name__ == "__main__":
    main()
