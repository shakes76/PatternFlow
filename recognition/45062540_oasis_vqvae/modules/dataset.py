import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import os

def load_data():
    """
    Store the file paths of the images from the specified directory
    
    Params: None
        
    Returns:
        Three lists containing file paths of all the images in the train, validate and test dataset respectively
    """

    #store the filenames of the train dataset
    data_train = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_train"):
        image_id = filename[5:]
        data_train.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_train", filename))

    #store the filenames of the validation dataset
    data_validate = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_validate"):
        image_id = filename[5:]
        data_validate.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_validate", filename))

    #store the filenames of the testing dataset
    data_test = []
    for filename in os.listdir("D:/keras_png_slices_data/keras_png_slices_test"):
        image_id = filename[5:]
        data_test.append(os.path.join("D:/keras_png_slices_data/keras_png_slices_test", filename))
    
    return data_train, data_validate, data_test

def preprocess_image(img):
    """
    Preprocess the image into a specific format before giving to the VQ-VAE model to train
    
    Params:
        img: the image to preprocess
        
    Returns:
        the image array after preoprocessing
    """
    img = np.array(img).astype('float32')
    #scale image pixels in (0,1)
    img = img / 255
    #add one additional axis
    img = img[:, :, np.newaxis]
    return img

def data_generator(train_data, batch_size = 8):
    """
    A generator outputs batch_size number of randomly selected preprocessed images from the training dataset
    
    Params:
        train_data: a list containing file paths of all the images in the training dataset
        batch_size: the number of images to be selected from the training dataset, default value = 8
        
    Returns:
        a list of batch_size number of preprocessed images randomly selected from the training dataset
    """
    while True:
        xs = []
        for i in range(batch_size):
            #randomly pick an image from the training datset
            img = random.choice(train_data)
            img = Image.open(img)
            #preprocess the image
            img = preprocess_image(img)
            xs.append(img)
        xs = np.array(xs).astype('float32')
        yield xs

def validate_generator(validate_data, batch_size = 8):
    """
    A generator outputs batch_size number of preprocessed images from the validate dataset in sequential order
    
    Params:
        validate_data: a list containing file paths of all the images in the validate dataset
        batch_size: the number of images to be selected from the validate dataset, default value = 8
        
    Returns:
        a list of batch_size number of preprocessed images selected from the validate dataset
    """
    count = 0
    while True:
        xs = []
        for i in range(batch_size):
            if count == len(validate_data):
                count = 0
            img = validate_data[count]
            count += 1
            img = Image.open(img)
            img = preprocess_image(img)
            xs.append(img)
        xs = np.array(xs).astype('float32')
        yield xs

def train_codebook_generator(train_data, vqvae, batch_size = 32):
    """
    A generator outputs batch_size numer of preprocessed images from the training dataset in sequential order
    
    Params:
        train_data: a list containing file paths of all the images in the training dataset
        vqvae: the vqvae model trained
        batch_size: the number of images to be selected from the training dataset, default value = 32
        
    Returns:
        a list of preprocessed images generated using the images from the training dataset
    """
    count = 0
    while True:
        xs = []
        for i in range(batch_size):
            if count == len(train_data):
                count = 0
            img = train_data[count]
            count += 1
            img = Image.open(img)
            #format the image to be passed to the vqvae encoder
            img = preprocess_image(img)
            xs.append(img)

        xs = np.array(xs)
        #preprocess the data to its closest key (latent embedding vector) in the codebook (latent space).
        xs = process_data(xs, vqvae)
        
        yield xs, xs
        
def validate_codebook_generator(validate_data, vqvae, batch_size = 32):
    """
    A generator outputs batch_size numer of preprocessed images from the validation dataset in sequential order
    
    Params:
        validate_data: a list containing filenames of all the images in the validate dataset
        vqvae: the vqvae model trained
        batch_size: the number of images to be selected from the validate dataset, default value = 32
        
    Returns:
        a list of preprocessed images generated using the images from the vaidate dataset
    """
    count = 0
    while True:
        xs = []
        for i in range(batch_size):
            if count == len(validate_data):
                count = 0
            img = validate_data[count]
            count += 1
            img = Image.open(img)
            #format the image to be passed to the vqvae encoder
            img = preprocess_image(img)
            xs.append(img)
            
        xs = np.array(xs)
        #preprocess the data to its closest key (latent embedding vector) in the codebook (latent space).
        xs = process_data(xs, vqvae)
 
        yield xs, xs
        
def process_data(data, vqvae):
    """
    Preprocess data into a specific format before giving to the PixelCNN model to train
    
    Params:
        data: the data to preprocess (in batches of size->(batch_num, 256,256,1))
        vqvae: the vqvae model trained
    
    Returns:
        The data after preprocessing (map to its closest latent embedding vector in the latent space) in one-hot encoded format
    """
    #predict the output from the encoder
    encoder_out = vqvae.encoder.predict_on_batch(data)
 
    #map the output from encoder to its closest latent embedding vector in the latent space in one-hot encoded form
    flatten = tf.reshape(encoder_out, [-1, encoder_out.shape[-1]])
    indices = vqvae.vq_layer.get_code_indices(flatten)
    indices = tf.one_hot(indices, vqvae.vq_layer.num_embeddings)
    indices = tf.reshape(indices, encoder_out.shape[:-1] + (vqvae.vq_layer.num_embeddings,))
    return indices