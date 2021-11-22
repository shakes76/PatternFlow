import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from model import *

#Dice loss function
def loss_fn(y_true, y_pred):

    return 1-(tf.reduce_sum(y_true*y_pred)*2/(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)))


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
    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_UNCHANGED)
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

#Split the data into training and testing sets; validations sets are split during model.fit()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

print("Converting arrays to tensors...")
#Convert arrays to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)/255
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)/255
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)


#Add extra dimension needed for bgr input
Y_train = tf.expand_dims(Y_train,-1)
Y_test = tf.expand_dims(Y_test,-1)

#use model.py functions to build unet
model = unet_improved(height,width,3,1)

#compile using static learning_rate and custom dice loss function
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=loss_fn)

#fit model for 100 epochs for best results
unet_trained = model.fit(X_train, Y_train, epochs=100, batch_size=20, shuffle=True, validation_split=0.1)

#run test X set through predictions
predictions = model.predict(X_test)

#threshold predictions
match = tf.greater(predictions, 0.91)
#cast true and false values as 0's and 1's
white_class = tf.cast(match, dtype=tf.float32)

#swap 0--->1 1---->0 so we can measure the dice of the black class
black_class = (white_class - 1) * -1
black_class_gt = (Y_test - 1) * -1

white_dice_total = 0
black_dice_total = 0

#compute average dice for each of the classes
for i in range(match.shape[0]):
    dice_white = 1-loss_fn(Y_test[i],white_class[i])
    dice_black = 1-loss_fn(black_class_gt[i],black_class[i])
    white_dice_total += dice_white
    black_dice_total += dice_black
avg_white = white_dice_total/white_class.shape[0]
avg_black = black_dice_total/black_class.shape[0]
print("Average - Mole:",avg_white,"Skin:",avg_black,"number of observations:",match.shape[0])


#visualisation

index = 30

plt.imshow(X_test[index,:,:,0])
plt.show()
plt.imshow(Y_test[index,:,:,0],cmap="gray")
plt.show()
plt.imshow(match[index,:,:,0],cmap="gray")
plt.show()

