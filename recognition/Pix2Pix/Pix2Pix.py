from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pathlib
import glob


# In[3]:


#!pip install matplotlib
#!pip install tqdm
#!pip install scikit-image
#!pip install keras
#!pip install image


# In[26]:


batch_size = 32
EPOCHS = 100


# In[18]:


########################## Training Directory #####################################################
train_x_dir = "C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_seg_train/"
train_y_dir = "C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_train/"

########################### Validation Dorectory ######################################################

val_x_dir = "C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_seg_validate/"
val_y_dir = "C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_validate/"


# In[20]:


train_imgs = glob.glob(train_x_dir + '*.png')
train_labels = glob.glob(train_y_dir + '*.png')

val_imgs = glob.glob(val_x_dir + '*.png')
val_labels = glob.glob(val_y_dir + '*.png')


# In[21]:


# Create a tf.data.Dataset from the filenames (and labels).
train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))


# In[ ]:


# Write a function that converts a filename and a label to a pair of data arrays.
def map_fn(filename, label):
    # Load the raw data from the file as a string.
    img = tf.io.read_file(filename)
    label = tf.io.read_file(label)
    
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.image.decode_jpeg(img, channels=1) # channels=1 for greyscale
    label = tf.image.decode_jpeg(label, channels=1) # channels=1 for greyscale
    
    # Resize the image to the desired size.
    img = tf.image.resize(img, (256, 256))
    label = tf.image.resize(label, (256, 256))
    
    # casting image as float32.
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.float32)
    
    #Standardise values to be in the [-1, 1] range.
    img = (img / 127.5) - 1
    label = (label / 127.5) - 1
    
    return img, label


# In[25]:


##################### genrating pictures, shuffling and selecting batch images from training images. ##########################
train_ds = train_ds.map(map_fn)
# Make the dataset to be reshuffled each time it is iterated over.
# This is so that we get different batches for each epoch.
# For perfect shuffling, the buffer size needs to be greater than or equal to the size of the dataset.
train_ds = train_ds.shuffle(len(train_imgs))
train_ds = train_ds.batch(batch_size)

##################### genrating pictures and selecting batch images from validation images. ##########################
val_ds = val_ds.map(map_fn)
val_ds = val_ds.batch(batch_size)


# In[ ]:





# In[ ]:





# In[ ]:





# # Generator 

# In[ ]:


def conv2d_func(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


# In[ ]:


def generator(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    
    input_image = Input(shape=input_img.shape)
    # Contracting Path
    c1 = conv2d_func(input_image, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_func(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_func(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_func(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_func(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_func(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_func(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_func(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_func(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# ## Discriminator
# 
# 

# In[5]:


def downsample(filters, size, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)

    conv = Conv2D(filters=filters, kernel_size=(size,size) , strides=2, padding='same', kernel_initializer=init)

    if apply_batchnorm:
        conv = BatchNormalization()(conv)
  
    conv = LeakyReLU(alpha=0.1)(conv)

    return conv


# In[9]:


def discriminator(inp_img, gen_img):
    init = tf.random_normal_initializer(0., 0.02)

    inp_img_d = Input(shape=inp_img.shape)
    gen_img_d = Input(shape=gen_img.shape)

    concat_img = concatenate([inp_img_d, gen_img_d])

    d1 = downsample(64, 2, False)(concat_img)
    d2 = downsample(128, 2, False)(d1)
    d3 = downsample(256, 2, False)(d2)
    d4 = downsample(512, 2, False)(d3)

    d_final = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d4)
    disc_out = Activation('tanh')(d_final)  
    
    disc_model = Model([inp_img_d, gen_img_d], disc_out)
    
    return disc_model


# # Training Model

# In[ ]:


def discriminator_loss(disc_real_output, disc_fake_out):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = binary_cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# In[ ]:


def generator_loss(disc_fake_out, fake_out, target):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = binary_cross_entropy(tf.ones_like(disc_fake_out), disc_fake_out)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# In[ ]:


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# In[10]:


def train_step(inp_img, real_img):

    fake_output = generator(inp_img, training=True)
  
    # Training discriminator
    with tf.GradientTape() as descr_tape:
    disc_real_out = discriminator([inp_img, real_img], training=True)
    disc_fake_out = discriminator([inp_img, fake_output], training=True)

    disc_loss = discriminator_loss(disc_real_out, disc_fake_out)

    discriminator_gradients = descr_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
    # Training Generator
    with tf.GradientTape() as gen_tape:
    disc_fake_out_decision = discriminator([inp_img, fake_output])

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_fake_out_decision, fake_output, target)
    
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables) 
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  


# In[ ]:


def train(train_ds, val_ds, EPOCHS):
    for epoch in range(EPOCHS):
        
        for n, (input_image, label) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, label)
            print()
    


# In[ ]:


train(train_ds, val_ds, EPOCHS)

