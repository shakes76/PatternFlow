#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageOps 
import image
import numpy as np
from matplotlib import pyplot as plt
import os
import tqdm
from tqdm import tqdm_notebook, tnrange
from skimage.transform import resize
import datetime

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time


def dataPreprocessing(batch_size):
    '''This function parses training and validation dataset from given directory 
    & preprocess it for model training'''
    
    
    print("Data preprocessing opereation is being running...")
    print("")
    
    imgName_X_train = next(os.walk("/keras_png_slices_data/keras_png_slices_seg_train"))[2] # list of names all images in the given path
    imgName_y_train = next(os.walk("/keras_png_slices_data/keras_png_slices_train"))[2] # list of names all images in the given path

    print("No. of training images = ", len(imgName_X_train))
    print("No. of training images labels = ", len(imgName_y_train))

    print ("")

    imgName_X_validate = next(os.walk("/keras_png_slices_data/keras_png_slices_seg_validate"))[2] # list of names all images in the given path
    imgName_y_validate = next(os.walk("/keras_png_slices_data/keras_png_slices_validate"))[2] # list of names all images in the given path

    print("No. of validating images = ", len(imgName_X_validate))
    print("No. of validating images labels = ", len(imgName_y_validate))    
    
    ################## generating numpy array of same size as training & validation dataset ############
    X_train = np.zeros((len(imgName_X_train), 256, 256, 1), dtype=np.float32)
    y_train = np.zeros((len(imgName_y_train), 256, 256, 1), dtype=np.float32)

    X_validate = np.zeros((len(imgName_X_validate), 256, 256, 1), dtype=np.float32)
    y_validate = np.zeros((len(imgName_y_validate), 256, 256, 1), dtype=np.float32)
    ####################################################################################################
    
    
    ############################################# For Training #######################################################
    # tqdm is used to display the progress bar
    for n_train, id_train in tqdm_notebook(enumerate(imgName_X_train), total=len(imgName_X_train)):
        # Loading training images
        img_train = load_img("/keras_png_slices_data/keras_png_slices_seg_train/"+id_train, grayscale=True)
        x_img_train = img_to_array(img_train)
        x_img_train = resize(x_img_train, (256, 256, 1), mode = 'constant', preserve_range = True)
        
        X_train[n_train] = (x_img_train / 127.5) - 1  ### making pixel values between -1 & 1
    
    for n_mask_train, id_mask_train in tqdm_notebook(enumerate(imgName_y_train), total=len(imgName_y_train)):
        # Loading training images
        mask_train = load_img("/keras_png_slices_data/keras_png_slices_train/"+id_mask_train, grayscale=True)
        y_img_train = img_to_array(mask_train)
        y_img_train = resize(y_img_train, (256, 256, 1), mode = 'constant', preserve_range = True)
        
        # Save images
        y_train[n_mask_train] = (y_img_train / 127.5) - 1  ### making pixel values between -1 & 1
    

    ########################## For Validation #######################################################
    for n_validate, id_validate in tqdm_notebook(enumerate(imgName_X_validate), total=len(imgName_X_validate)):
        # Loading validating images
        img_validate = load_img("/keras_png_slices_data/keras_png_slices_seg_validate/"+id_validate, grayscale=True)
        x_img_validate = img_to_array(img_validate)
        x_img_validate = resize(x_img_validate, (256, 256, 1), mode = 'constant', preserve_range = True)
        
        X_validate[n_validate] = (x_img_validate / 127.5) - 1  ### making pixel values between -1 & 1
    
    for n_mask_validate, id_mask_validate in tqdm_notebook(enumerate(imgName_y_validate), total=len(imgName_y_validate)):
        # Loading validating images
        mask_validate = load_img("/keras_png_slices_data/keras_png_slices_validate/"+id_mask_validate, grayscale=True)
        y_img_validate = img_to_array(mask_validate)
        y_img_validate = resize(y_img_validate, (256, 256, 1), mode = 'constant', preserve_range = True)
        
        # Save images
        y_validate[n_mask_validate] = (y_img_validate / 127.5) - 1  ### making pixel values between -1 & 1
        
        
    # Creating a tf.data.Dataset from above created input & labels images.
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    
    
    ##################### Shuffling and selecting batch images from training dataset. ##########################
    train_ds = (train_ds.batch(batch_size).shuffle(len(imgName_X_train)))
    
    ##################### Selecting batch images from validation dataset. ##########################
    val_ds = val_ds.batch(batch_size)
    
    
    return train_ds, val_ds


## Generator 

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


def make_generator(n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function defining the UNET Model"""
    
    input_image = Input(shape=(256, 256, 1))
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
    
    outputs = Conv2D(1, (1, 1), activation='tanh')(c9)
    
    model = Model(inputs=input_image, outputs=outputs)
    return model


generator = make_generator() # making a generator constructor


# ## Discriminator

def downsample(layer, filters, size, apply_batchnorm=True):
    """This function just performs convolution operation with the parameters passed to it"""
    
    init = tf.random_normal_initializer(0., 0.02)

    conv = Conv2D(filters=filters, kernel_size=(size,size) , strides=2, padding='same', kernel_initializer=init)(layer)

    if apply_batchnorm:
        conv = BatchNormalization()(conv)
  
    conv = LeakyReLU(alpha=0.2)(conv)

    return conv


def make_discriminator():
    """Function used to discriminate the real & fake images generated from generator"""
    
    init = tf.random_normal_initializer(0., 0.02)

    inp_img_d = Input(shape=(256, 256, 1))
    gen_img_d = Input(shape=(256, 256, 1))

    concat_img = concatenate([inp_img_d, gen_img_d])

    d1 = downsample(concat_img, 64, 4, True)
    d2 = downsample(d1, 128, 4, True)
    d3 = downsample(d2, 256, 4, True)
    d4 = downsample(d3, 512, 4, True)

    d_final = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d4)
    disc_out = Activation('sigmoid')(d_final)  
    
    disc_model = Model(inputs=[inp_img_d, gen_img_d], outputs=disc_out)
    
    return disc_model


discriminator = make_discriminator() # making a discriminator constructor


# # Training Model


def discriminator_loss(disc_real_output, disc_fake_out):
    """This function calculates the real loss, generated/fake loss & total discriminant loss"""
    
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = binary_cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = binary_cross_entropy(tf.zeros_like(disc_fake_out), disc_fake_out)
    
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_fake_out, fake_out, real_img):
    """This function calculates the Mean absolute error & gan loss"""
    
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = binary_cross_entropy(tf.ones_like(disc_fake_out), disc_fake_out)

    # mean absolute error
    mae = tf.reduce_mean(tf.abs(real_img - fake_out))

    #total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    total_gen_loss = gan_loss + (100*mae)
    
    return total_gen_loss, gan_loss, mae


"""initialization of generator & discriminator optimisers"""
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


"""Below code creates a log file which is used later to store losses while model is under training"""
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def train_step(inp_img, real_img, epoch):
    """
    This function using gradient tape trains generator and discriminator models.
    """
    
    # Training discriminator & generator
    with tf.GradientTape() as descr_tape, tf.GradientTape() as gen_tape:
        fake_output = generator(inp_img, training=True)
        
        disc_real_out = discriminator([inp_img, real_img], training=True)
        disc_fake_out = discriminator([inp_img, fake_output], training=True)

        disc_loss = discriminator_loss(disc_real_out, disc_fake_out)
        gen_total_loss, gen_gan_loss, gen_mae = generator_loss(disc_fake_out, fake_output, real_img)
        
    discriminator_gradients = descr_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)    
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    
    
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables) 
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))    
    
    """
    Below set of code writes generator total loss, gan loss, mean absolute error and discrimintor loss on 
    summary file used to track mentioned losses on tensorboard. 
    """ 
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_mae', gen_mae, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        
    return gen_total_loss, gen_gan_loss, gen_mae, disc_loss



def generate_images(test_input, tar):
    """
    This function genrates comapre real image with generated image.
    """
    prediction = generator(test_input, training=True) #predicting image using generator and input image.
    
    plt.figure(figsize=(20,20))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5) # making pixels values between 0 and 1 for symmetric visualization.
        plt.axis('off')
    plt.show()


def train(train_ds, val_ds, EPOCHS):
    """
    This function calls the train_step function and train model on a set of image batches at a time.
    Moreover, after each epoch an image from validation dataset is pushed through generator to check model's performce on validation dataset.
    """
    for epoch in range(EPOCHS):
        
        print("Training Epoch " + str(epoch+1) + ".........")
            
        for n, (input_image, label) in tqdm.notebook.tqdm(train_ds.enumerate()):
            train_step(input_image, label, epoch)
        
        print("Result after " + str(epoch+1) + " epoch")
        for inp, tar in val_ds.take(1):
            generate_images(inp, tar)


def run_Pix2Pix(batch_size, EPOCHS):
    """
    This function call the dataPreprocessing function to do data preprocessing followed by triggering model training function.
    """
    train_ds, val_ds = dataPreprocessing(batch_size)
    
    start_time = time.time()
    train(train_ds, val_ds, EPOCHS)
    end_time = time.time()

    print("Total training time: " + str(end_time - start_time) + " seconds.")

