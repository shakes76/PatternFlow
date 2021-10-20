import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Concatenate, Conv2DTranspose, Reshape, Permute, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


X = [] #list to store the input images for cnn
Y = [] #list to store the ground truth to comapre the output of the cnn to.

root = "C:/Users/s4630726/Downloads" #directory where ISIC data is stored


path = os.path.join(root, "ISIC2018_Task1-2_Training_Input_x2") 
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    X.append(img_array)
print(img)
plt.imshow(img_array,cmap="gray")
plt.show()

path = os.path.join(root, "ISIC2018_Task1_Training_GroundTruth_x2") 
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    Y.append(img_array)
print(img)
plt.imshow(img_array,cmap="gray")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #split the data into training and testing sets validations sets made in model.fit

