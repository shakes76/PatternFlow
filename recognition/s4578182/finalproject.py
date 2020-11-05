#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# In[16]:


ground=glob.glob("C:/Users/s4578182/Desktop/ISIC2018_Task1_Training_GroundTruth_x2/*.png")
train=glob.glob("C:/Users/s4578182/Desktop/ISIC2018_Task1-2_Training_Input_x2/*.jpg")


# In[17]:


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras


# In[18]:


print('Size of training set:', len(train))


# In[19]:


print('Size of training set:', len(ground))


# In[ ]:




