#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt



# In[8]:


def load_img_from_dir(data_dir, img_format = 'png', imagecount = 0, tensor = True):
    '''
    Helper function to load data from the data directory.
    
    Params:
    
    data_dir: Folder Path/ Data Directory.
    imagecount: number of images to be loaded. Default = complete dataset.
    tensor: format of image. Default Tensor. If False then images will be loaded as numpy ndarray.
        
    '''
    from PIL import Image
    import glob
    import tensorflow as tf
    import numpy as np
    
    image_list = []
    if imagecount !=0:
        for file in glob.glob(data_dir+'\*.'+img_format): 
            im = np.asarray(Image.open(file))
            im = np.expand_dims(im, axis=2)
            if tensor == True:
                im = tf.image.convert_image_dtype(
                im, tf.int64, saturate=False, name=None)
                im = tf.keras.preprocessing.image.img_to_array(im)

            
            image_list.append(im)
            if len(image_list)==imagecount:
                break
    else:
        for file in glob.glob(data_dir+'\*.'+img_format): 
            im=Image.open(file)
            im = tf.image.convert_image_dtype(
                np.asarray(im), tf.int64, saturate=False, name=None)
            if tensor == True:
                im = tf.image.convert_image_dtype(
                im, tf.int64, saturate=False, name=None)
            
            
            
            image_list.append(im)
            
    return np.asarray(image_list)
    


def tf_plot_gallery(images, h, w,titles = 0, n_row=3, n_col=4):
    """Helper function to plot a gallery of tensor portraits"""
    
    import tensorflow as tf
    if titles == 0:
        titles = np.arange(len(images))
    
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(tf.reshape(images[i],[h, w]), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# In[ ]:



def plot_gallery(images,  h, w, titles = 0, n_row=3, n_col=4):
    """Helper function to plot a gallery of all format portraits"""
    
    import numpy as np
    
    if titles == 0:
        titles = np.arange(len(images))
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(np.asarray(images[i]).reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



