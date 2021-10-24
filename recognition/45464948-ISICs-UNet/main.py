from preprocesseddata import *
import glob
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split


# load image ISIC image
isic_input = glob.glob("D:/2021S2/COMP3710/ass/report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg")
isic_ground_truth = glob.glob("D:/2021S2/COMP3710/ass/report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png")


#use mnist first
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

X = preprocess_array(isic_input)/ 255
y = np.round(preprocess_array_truth(isic_ground_truth)/ 255)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=66)