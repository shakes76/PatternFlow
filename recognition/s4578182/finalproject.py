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


# In[51]:


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


# In[ ]:





# In[ ]:




