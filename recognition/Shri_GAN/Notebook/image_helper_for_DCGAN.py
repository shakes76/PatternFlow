#!/usr/bin/env python
# coding: utf-8

# In[ ]:

















# In[1]:


def load_img_from_dir(data_dir, img_size = (128,128), img_format = 'png', imagecount = 0, tensor = True):
    '''
    Helper function to load data from the data directory.
    
    Params:
    
    data_dir: Folder Path/ Data Directory.
    imagecount: number of images to be loaded. Default = complete dataset.
    tensor: format of image. Default Tensor. If False then images will be loaded as numpy ndarray.
        
    '''
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import os, time  
    import numpy as np 
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    import tensorflow as tf 
    from PIL import Image
    import glob
    import tensorflow as tf
    import numpy as np
    
    image_list = []
    if imagecount !=0:
        for file in glob.glob(data_dir+'\*.'+img_format): 
            im = Image.open(file)
            im = im.resize(img_size) 
            im = np.asarray(im)
            im = np.expand_dims(im, axis=2)
            
            if tensor == True:
                im = tf.image.convert_image_dtype(
                im, tf.float32, saturate=False, name=None)
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
            
    return image_list
    


# In[2]:


def tf_load_img_from_dir(dir_data, img_shape = (128, 128, 1), img_format = 'png', imagecount = 0, tensor = True):
    '''
    Helper function to load data from the data directory.
    
    Params:
    
    data_dir: Folder Path/ Data Directory.
    imagecount: number of images to be loaded. Default = complete dataset.
    tensor: format of image. Default Tensor. If False then images will be loaded as numpy ndarray.
        
    '''
    
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import os, time  
    import numpy as np 
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    import tensorflow as tf 
    from PIL import Image
    import glob
    import tensorflow as tf
    import numpy as np
    
    Ntrain = imagecount
    nm_imgs       = np.sort(os.listdir(dir_data))
    nm_imgs_train = nm_imgs[:Ntrain]
    X_train = []
    for i, myid in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
        im = img_to_array(image)
        im = tf.image.convert_image_dtype(im, tf.float32, saturate=False, name=None)
        im = tf.keras.preprocessing.image.img_to_array(im)
        image = tf.image.rgb_to_grayscale(im)
        X_train.append(image)
    X_train = tf.convert_to_tensor(X_train)
    return(X_train)
   



# In[3]:



def tf_plot_gallery(images, img_shape, titles = 0, n_row=3, n_col=4):
    
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import os, time  
    import numpy as np 
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    import tensorflow as tf 
    from PIL import Image
    import glob
    import tensorflow as tf
    import numpy as np
    
    """Helper function to plot a gallery of tensor portraits"""
    
    h = img_shape[0]
    w = img_shape[1]
    t = img_shape[2]
    import tensorflow as tf
    if titles == 0:
        titles = np.arange(len(images))
    
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(tf.reshape(images[i],[h, w, t]), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



# In[4]:



def plot_gallery(images,  h, w, titles = 0, n_row=3, n_col=4):
    
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import os
    import matplotlib.pyplot as plt
    import os, time  
    import numpy as np 
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    import tensorflow as tf 
    from PIL import Image
    import glob
    import tensorflow as tf
    import numpy as np
    
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



# In[5]:


def cv_get_images(data_dir, img_format = '.png', img_size = (128,128), gray = True, tensor = True ):
    """
    Helper function to get images from local directory using OpenCV library.
    
    input params:
    
    data_dir: local directory for image data.
    img_format: format of the images. default = .png
    img_size: size of the image, default = 128x128
    
    gray: Convert to Grayscale if True.
    tensor: Convert to TF structure if true.
    
    """
    import tensorflow as tf
    #print(tf.__version__)
    import glob
    import cv2
    from IPython import display
    
    
    import matplotlib.pyplot as plt
    import os, time  
    import numpy as np 
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    
    
    images_train = [cv2.imread(file) for file in glob.glob(data_dir + "//*"+img_format)]
    
    if gray ==True:
        # converting RGB to grayscale
        X_train = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_train]

        #resizing images to 128 x 128 
        X_train = [cv2.resize(image, (128,128)) for image in X_train]
    
    if tensor==True:
    
        #converting to tensors
        tf_X_train = tf.convert_to_tensor(X_train, dtype = tf.float32)
        #print(tf_X_train.shape)
        
        tf_X_train =  tf.reshape(tf_X_train, [tf_X_train.shape[0], img_size[0], img_size[1],1])
        #print('max = ', tf.reduce_max(tf_X_train).numpy())
        #print('min = ', tf.reduce_min(tf_X_train).numpy())
        
    if gray == True:
        #Normalising the dataset 
        tf_X_train = (tf_X_train - 127.5)/127.5
        
    return tf_X_train
    
    



# In[10]:


#Test script for cv_get_images
#data_dir = "H://PatternLab//PatternRecognition//Datasets//brain//keras_png_slices_data//keras_png_slices_train//path_test"
#tf_X_train = cv_get_images(data_dir)
#tf_X_train.shape


# In[7]:


# Test script for tf_plot_gallery
#tf_plot_gallery(tf_X_train,(128,128,1))

