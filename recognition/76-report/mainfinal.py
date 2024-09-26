#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import random
import warnings
from tensorflow.keras.models import Model, load_model

import sys
sys.path.append('C:/Users/s4547877/PatternFlow/recognition/76-report/')
from im_unet import  *
from load import *
from dice import *

os.chdir('C:/Users/s4547877/Downloads/')


# In[4]:


#get path
input_PATH='./ISIC2018_Task1-2_Training_Input_x2/'
input_ids = next(os.walk(input_PATH))[2]  
truth_PATH='./ISIC2018_Task1_Training_GroundTruth_x2/'
truth_ids = next(os.walk(truth_PATH))[2] 


# In[5]:


X_train,X_val,X_test,y_train,y_val,y_test=loaddata(input_PATH,input_ids,truth_PATH,truth_ids)


# In[8]:


#show a sample
import matplotlib.pyplot as plt
img = imread(input_PATH + input_ids[1])
#img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
print(img.shape)
plt.imshow(img)
plt.show()


# In[6]:


model=unet()


# In[7]:


result=model.fit(X_train,y_train,batch_size=2,validation_data=(X_val,y_val),epochs=20)


# In[ ]:


#visiualize training procedure
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(result.history['loss'],label='loss')
plt.plot(result.history['val_loss'],label='val loss')
plt.legend()
plt.subplot(2,2,2)
plt.plot(result.history['dice_coef'],label='dice_coef')
plt.plot(result.history['val_dice_coef'],label='val dice_coef')
plt.legend()


# In[ ]:


#test set prediction
y_test_pred=model.predict(X_test)


# In[ ]:


#show the first test set and its prediction
re = model.predict(X_test)

re = re > 0.5
fig = plt.figure(figsize = (16,8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 3, 1)
ax.imshow(X_test[0])
plt.title('X_test')
ax = fig.add_subplot(1, 3, 2)
ax.imshow(np.reshape(y_test[0]*255, (256,256)), cmap="gray")
plt.title('y_test')
ax = fig.add_subplot(1, 3, 3)
ax.imshow(np.reshape(re[0]*255, (256,256)), cmap="gray")
plt.title('y_test_prediction')


# In[ ]:


#dice coefficient of the testset 
scores = model.evaluate(X_test,y_test,batch_size=16)
print("dice_coef = ", scores[1])

