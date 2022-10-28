#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# We need to import our load_model function 
from tensorflow.keras.models import load_model
import keras
from dataset import *
import matplotlib.pyplot as pyplot

model = load_model('improved_UNET.h5')


# In[ ]:


model.evaluate(np.round(model.predict(x_test,batch_size=8)))


# In[ ]:


pred_image = np.round(model.predict(x_test,batch_size=8))


# In[ ]:


def imshow(title, image = None, size = 20):
    if image.any():
        w, h = image.shape[0], image.shape[1]
        aspect_ratio = w/h
        plt.figure(figsize=(size * aspect_ratio,size))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    else:
        print("Image not found")


# In[ ]:


for i in range(0,6):
    random_num = np.random.randint(0, len(masks_test_images))
    img = masks_test_images[random_num]
    img1 = pred_image[random_num]
    img_concate_Hori=np.concatenate((img,img1),axis=1)
    imshow("Ground Truth & prediction",img_concate_Hori , size = 2)

