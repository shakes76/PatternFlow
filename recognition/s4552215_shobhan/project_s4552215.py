#HUMAN BRAIN MRI IMAGE SEGMENTATION with UNET Demo/prac2
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
# In[9]:


# Initialising the compression dimensions
img_width = 128
img_height = 128
border = 5



# Extracting the file names of the images and masks in training, test and validation folders 
ids_train = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_train"))[2] 
ids_test = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_test"))[2]
ids_val = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_validate"))[2]
ids_seg_train = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_seg_train"))[2]
ids_seg_test = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_seg_test"))[2]
ids_seg_val = next(os.walk("H:/svn/keras_png_slices_data/keras_png_slices_seg_validate"))[2]
print("No. of images in training folder= ", len(ids_train))
print("No. of images in test folder= ", len(ids_test))
print("No. of images in validation folder= ", len(ids_val))


# #SELFSTUDY:
# #numpy. zero return a new array of given shape and type, filled with zeros.
# #for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):https://stackoverflow.com/questions/49642866/python-for-loop-multiple-values-objects-after-in : You can ignore the tqdm function because if you were to remove it the only thing that would be different is you wouldn't see a load bar but everything else would work just fine. This code
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
# works the same as just without a load bar.
# for n, id_ in enumerate(test_ids):
# what enumerate does is places the index number of each item in a tuple that is given in a new dimension of the array. So if the array input was 1D then the output array would be 2D with one dimension the indexes and the other the values. lets say your test_ids are [(2321),(2324),(23213)] now with enumerate the list now looks like [(0,2321),(1,2324),(2,23213)] what putting a comma does is once given a value (from our case the for loop) say (0,2321) is separate each tuple value in the order they are given so in this case they would equal n = 0 and id_ = 2321

# In[21]:


# Function for loading images from the folders
def loading_img(inp_path,ids):
    X = np.zeros((len(ids), img_height, img_width, 1), dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)): 
        # Load images
        img = load_img(inp_path+id_, color_mode = 'grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
        X[n] = x_img/255
        
    return X


# In[22]:


def loading_seg(inp_path,ids):
    X = np.zeros((len(ids), img_height, img_width, 1), dtype=np.uint8)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(inp_path+id_, color_mode = 'grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
        X[n] = x_img
        
    return X


# In[23]:


# Loading the images and the masks for the training data set
XOasis_train= loading_img("H:/svn/keras_png_slices_data/keras_png_slices_train/",ids_train)


# In[24]:


print("No. of images in XOasis training folder= ", len(XOasis_train))


# In[25]:


yOasis_train= loading_seg("H:/svn/keras_png_slices_data/keras_png_slices_seg_train/",ids_seg_train)   
print("No. of images in yOasis training folder= ", len(yOasis_train))


# In[26]:


# Loading the images and the masks for the test data set
XOasis_test= loading_img("H:/svn/keras_png_slices_data/keras_png_slices_test/",ids_test)
yOasis_test = loading_seg("H:/svn/keras_png_slices_data/keras_png_slices_seg_test/",ids_seg_test)


# In[27]:


# Loading the images and the masks for the validation data set
XOasis_val= loading_img("H:/svn/keras_png_slices_data/keras_png_slices_validate/",ids_val)
yOasis_val = loading_seg("H:/svn/keras_png_slices_data/keras_png_slices_seg_validate/",ids_seg_val)


# SelfSTUDY:
# #In Python 3, they made the / operator do a floating-point division, and added the // operator to do integer division (i.e. quotient without remainder)
# 
# 255(0 to 255 pix for grey) / 85 = 3; 85 * 3 =255. 3 as black,white grey layers

# In[28]:


yOasis_train_sc = yOasis_train//85
yOasis_test_sc = yOasis_test//85
yOasis_val_sc = yOasis_val//85


# In[29]:


yOasis_test_sc


# In[30]:


yOasis_train_cat = to_categorical(yOasis_train_sc)
yOasis_test_cat = to_categorical(yOasis_test_sc)
yOasis_val_cat = to_categorical(yOasis_val_sc)


# In[31]:


yOasis_train_cat


# In[1]:


# Build the UNET model

#input(128*128*1[number of channel])->2 layers of C1(128*128*16[number of filters])->P1(64*64*32)-> 2 Layers of c2(64*64*32)

#padding=same(we want output image as the same dimension of input image)
#Conv=3*3 ReLU
#Maxpool=2*2 ;stride=2
#Upsample=2*2
#Final Conv=1*1
#Dropout betwn conv steps to prevent overfitting


# In[6]:


#Layers module in Keras deals with Conv,recurring,maxpooling,Normalaization etc
#Input layer
inputs = tf.keras.layers.Input((128,128,1)) 


# In[17]:


#convolution layers
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
#Converting all inputs int to floating;lambda is a python function

##contraction path:
c1 =tf.keras.layers.Conv2D(16,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(s)
#feature dimension=16,kernel size=3,3; 
#deep neural network always update the weights so we need a strating weight to get started,
#from there kernel initializer comes up,which get updated.
#he_normal = trunkated normal distribution(centered around zero,doesnot go far like normal !! distribution!)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 =tf.keras.layers.Conv2D(16,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(c1)
#dropping 10% of c1 to reduce overfitting
p1=tf.keras.layers.MaxPooling2D((2,2))(c1)


c2 =tf.keras.layers.Conv2D(32,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 =tf.keras.layers.Conv2D(32,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(c2)
p2=tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 =tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 =tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(c3)
p3=tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 =tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 =tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(c4)
p4=tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 =tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 =tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)


##EXPANSIVE PATH
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(4,(1,1), activation='softmax')(c9)
model= tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer = 'adam', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

#optimizer is a module that contain lot of backpropagation algoritham that can train our model.Optimizer tries to minimize the loss function.When optimizer finds minimum loss function it stop iteration.


# In[32]:


# Providing conditions for the training to stop based on validation loss
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-OASIS.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# In[33]:


results = model.fit(XOasis_train, yOasis_train_cat, batch_size=32, epochs=20, callbacks=callbacks,validation_data=(XOasis_val, yOasis_val_cat))


# In[34]:


# Plotting the training and validation loss with respect to epochs
plt.figure(figsize=(8, 8))
plt.title("Cross Entropy Loss")
plt.plot(results.history["loss"], label="training_loss")
plt.plot(results.history["val_loss"], label="validation_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.legend();


# In[35]:


# Plotting the training and validation accuracy with respect to epochs
plt.figure(figsize=(8, 8))
plt.title("Classification Accuracy")
plt.plot(results.history["accuracy"], label="training_accuracy")
plt.plot(results.history["val_accuracy"], label="validation_accuracy")
plt.plot( np.argmin(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.legend();


# In[36]:


# load the best model
model.load_weights('model-OASIS.h5')


# In[37]:


test_preds = model.predict(XOasis_test, verbose=1)


# In[38]:


# Dice Coeffient
from keras import backend as K
def dice_coeff(y_true, y_pred, smooth=1):
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis=0)
    return coeff_dice


# In[39]:


# Printing Dice Coefficient
dice_coeff(yOasis_test_cat, test_preds, smooth=1)


# In[40]:


def plot_Oasis(X, y, y_pred, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix

    

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].contour(X[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Actual Image')

    ax[1].imshow(y[ix,...,0],cmap='gray')
    ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Actual Segment Image')

    ax[2].imshow(y_pred[ix,...,0],cmap='gray')
    ax[2].contour(y_pred[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted Segment Image')


# In[41]:


test_preds_arg = np.argmax(test_preds, axis = -1)*85


# In[42]:


n,h,w,g = yOasis_test.shape


# In[43]:


n,h,w,g


# In[45]:


test_preds_reshape = test_preds_arg.reshape(n,h,w,g)


# In[46]:


test_preds_reshape.shape


# In[49]:


plot_Oasis(XOasis_test,yOasis_test,test_preds_reshape)

