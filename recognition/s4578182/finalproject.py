#!/usr/bin/env python
# coding: utf-8

# In[49]:


import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# In[50]:


ground=glob.glob("C:/Users/s4578182/Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png")
train=glob.glob("C:/Users/s4578182/Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg")


# In[103]:


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras


# In[52]:


print('Size of training set:', len(train))


# In[53]:


print('Size of ground set:', len(ground))


# In[71]:


ground_images = []

for dire in range(len(ground)):
    img =np.array((Image.open(ground[dire]).resize((256, 256))))
    ground_images.append(img)


# In[74]:


train_images = []

for dire in range(len(train)):
    img = np.array((Image.open(train[dire]).resize((256, 256))))
    train_images.append(img)


# In[70]:


plt.imshow(test)


# In[59]:


plt.imshow(new1)


# In[72]:


for dir1 in range(len(ground_images)):
    new1=ground_images[dir1]
    for i in range(len(new1)):
        for j in range(len(new1[i])):
            if(new1[i][j]>0 and new1[i][j]<255):
                new1[i][j]=127
    ground_images[dir1]=new1
           


# In[89]:


train_images=np.array(train_images)
train_images.shape


# In[78]:


ground_img=np.expand_dims(np.array(ground_images),-1)
print(ground_img.shape)


# In[91]:


ground1=tf.data.Dataset.from_tensor_slices(ground_img)


# In[90]:


train1= tf.data.Dataset.from_tensor_slices(train_images)


# In[104]:


inputs = Input((256,256,3))
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
model.summary()


# In[106]:


history = model.fit(train1, validation_data=ground1, epochs=10, verbose=0)


# In[ ]:




