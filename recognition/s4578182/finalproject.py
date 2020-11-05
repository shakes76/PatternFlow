#!/usr/bin/env python
# coding: utf-8

# In[49]:


import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:





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


# In[54]:


print(train[0])


# In[69]:


test = np.array((Image.open(ground[0]).resize((256, 256))))
print(test.shape)
print(ground[0])
print(np.unique(test))


# In[71]:


ground_images = []

for dire in range(len(ground)):
    img =np.array((Image.open(ground[dire]).resize((256, 256))))
    ground_images.append(img)


# In[57]:


print(len(ground_images))


# In[ ]:


train_images = []

for directory in train:
    img = np.array(Image.open(train[0]))
    train.append(img)


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
           


# In[ ]:





# Do following to reduce classes:
# 
# 
# 1. keep 0 as 0.
# 2. keep 255 as 255.
# 3. anythin in between, make it 127.
# 
# 

# In[99]:


X=[]
for i in range(len(ground)):
    y=np.array((Image.open(ground[i]).resize((256, 256))))
    X.append(y)


# In[113]:


classes = train.classes
num_classes = len(train.classes)


# In[97]:


print(train[0])


# In[135]:


import matplotlib.image as mpimg
img = mpimg.imread(ground[100])
imgplot = plt.imshow(img)
plt.show()


# In[126]:


print(len(np.unique(X)))


# In[109]:


print(np.unique(X2))


# In[ ]:





# In[83]:


X=np.expand_dims(np.array(X),-1)
print(X.shape)


# In[87]:


X1 = tf.data.Dataset.from_tensor_slices(X)


# In[96]:


X2=[]
for i in range(len(train)):
    y1=np.array((Image.open(train[i]).resize((256, 256))))
    X2.append(y1)


# In[ ]:


print()


# In[86]:


X5 = tf.data.Dataset.from_tensor_slices(X2)


# In[76]:


X1=[]
for i in range(len(ground)):
    x1=np.array((Image.open(train[i]).resize((400, 400))))
    X1.append(x1)


# In[ ]:


X1 = tf.data.Dataset.from_tensor_slices(X1)


# In[79]:


type(X1)


# In[ ]:





# In[80]:


inputs = Input((400,400,1))
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))
merge6 = concatenate([conv4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
conv10 = Conv2D(NUM_CLASSES, 1, activation = 'softmax')(conv9)

model = Model(input = inputs, output = conv10)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'sparse_categorical_cross_entropy', metrics = ['accuracy'])
    
model.summary()


# In[ ]:


history = model.fit(X_tr, y_tr, epochs=10, validation data=


# In[90]:


train_image_names = pd.Series(train)


# In[ ]:





# In[ ]:




