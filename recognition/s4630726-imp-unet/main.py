import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from model import *

#Defining dimensions for cnn input and resizing
width = 256 
height = 192
resize_dim = (height, width) 

#List to store the input images for cnn
X = [] 
#List to store the ground truth to comapre the output of the cnn to.
Y = [] 

#Directory where ISIC data is stored
root = "C:/Users/s4630726/Downloads" 

print("Iterating through training input...")
#Iterate through folder reading each image and resizing them to a 4:3 aspect ratio of 256 x 192 then adding them to a global list
path = os.path.join(root, "ISIC2018_Task1-2_Training_Input_x2") 
for img in os.listdir(path): 
    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    img_array = resize(img_array, resize_dim, order=1, preserve_range=True, anti_aliasing=False).astype('uint8')
    X.append(img_array)



print("Iterating though groundtruth...")
path = os.path.join(root, "ISIC2018_Task1_Training_GroundTruth_x2") 
for img in os.listdir(path): 
    img_array_2 = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    #Resize anti_aliasing set to false so new pixel values aren't introduced in the segmentation map
    #The function also changes the values [0,255] to [0,1] which is critical for the umap segmentaiton to work correctly
    img_array_2 = resize(img_array_2, resize_dim, order=0, preserve_range=False, anti_aliasing=False).astype('uint8') 
    Y.append(img_array_2)


#print(img_array)
#print(img)                             @for debugging purposes
#plt.imshow(img_array,cmap="gray")
#plt.show()
#print(img_array_2)
#print(img)                             
#plt.imshow(img_array_2,cmap="gray")
#plt.show()


#Split the data into training and testing sets; validations sets are split during model.fit()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

print("Converting arrays to tensors...")
#Convert arrays to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.int16)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.int16)

#Add extra dimension needed for cnn input
X_train = tf.expand_dims(X_train,-1)
X_test = tf.expand_dims(X_test,-1)
Y_train = tf.expand_dims(Y_train,-1)
Y_test = tf.expand_dims(Y_test,-1)

model = unet_improved(height,width,2)

model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

unet_trained = model.fit(X_train, Y_train, epochs=20, batch_size=26, shuffle=True, validation_split=0.1)

predictions = model.predict(X_test)


#VISUALISATION OF RESULTS

#tf.print(predictions)

match = tf.math.argmax(predictions,axis=3)

#tf.print(match)

plt.imshow(X_test[0,:,:,0],cmap="gray")
plt.show()
plt.imshow(Y_test[0,:,:,0],cmap="gray")
plt.show()
plt.imshow(match[0,:,:],cmap="gray")
plt.show()

