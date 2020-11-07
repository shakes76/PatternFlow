import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense,Reshape,Dropout,LeakyReLU,Flatten,BatchNormalization,Conv2D,Conv2DTranspose,MaxPool2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


## Generator design
def buildGen(noise_size,ksize, strid):
    
    generator = Sequential()
    generator.add(Dense(8 * 8 * 64, activation="relu", input_shape=[noise_size]))
    generator.add(Reshape([8, 8, 64]))
    generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(256, kernel_size=ksize, strides=strid, padding="same", activation="relu"))
    generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(128, kernel_size=ksize, strides=strid, padding="same", activation="relu"))
    generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(32, kernel_size=ksize, strides=strid, padding="same", activation="relu"))
    generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(1, kernel_size=ksize, strides=strid, padding="same", activation="tanh"))
    generator.add(BatchNormalization())
    return generator


## Design Generator
def buildDis(ksize, strid, poolsize):
    discriminator = Sequential()
    discriminator.add(Conv2D(48, kernel_size=ksize, activation="relu", input_shape=[128, 128,1]))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization())
    discriminator.add(MaxPooling2D(pool_size=poolsize))
    discriminator.add(Conv2D(64, kernel_size=ksize, activation="relu"))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization())
    discriminator.add(MaxPooling2D(pool_size=poolsize))
    discriminator.add(Conv2D(128, kernel_size=ksize, activation="relu"))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization())
    discriminator.add(MaxPooling2D(pool_size=poolsize))
    discriminator.add(Dropout(0.25))
    discriminator.add(BatchNormalization())
    discriminator.add(Flatten())
    discriminator.add(Dense(25, activation="relu"))
    discriminator.add(Dense(1, activation="sigmoid"))
    return discriminator

## Build network
def buildGAN(generator, discriminator, learningr, beta_1v):
    opt = tf.keras.optimizers.Adam(learning_rate=learningr, beta_1=beta_1v)
    opt2 = tf.keras.optimizers.Adam(learning_rate=learningr, beta_1=beta_1v)
    GAN = Sequential([generator, discriminator])
    discriminator.compile(loss="binary_crossentropy", optimizer=opt2)
    discriminator.trainable = False
    GAN.compile(loss="binary_crossentropy", optimizer=opt)
    return GAN

def trainnetwork(GAN, epochs, batchsize, X_train, noise_size):
    ## Learning parameters and batch preparation
    epochs = epochs
    batch_size = batchsize
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    
    generator, discriminator = GAN.layers
    ## Learning loop
    for epoch in range(epochs):    
        print(f"Currently on Epoch {epoch+1}")
        i = 0
        # For every batch in the dataset
        for X_batch in dataset:
            i=i+1
        
            
            ## Discriminator Training            
            noise = tf.random.normal(shape=[batch_size, noise_size])
            gen_images = generator(noise)
            
            # Concatenate new images against the training set       
            X_fake_vs_real = tf.concat([gen_images,np.expand_dims(tf.dtypes.cast(X_batch,tf.float32),axis = 3)], axis=0)
            
            # Target tensor with 0 for fake, 1 for real
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            
            # This gets rid of a Keras warning
            discriminator.trainable = True
            
            # Train the discriminator on this batch
            discriminator.train_on_batch(X_fake_vs_real, y1)
            
            ## Generator training    
            
            # Create some noise
            noise = tf.random.normal(shape=[batch_size, noise_size])
            
            # We want discriminator to belive that fake images are real
            y2 = tf.constant([[1.]] * batch_size)
            
            # Avoids a warning
            discriminator.trainable = False
            
            GAN.train_on_batch(noise, y2)
        
        
        # Show progres images
        noise = tf.random.normal(shape=[5, noise_size])
        images = generator(noise)
        images = images.numpy()
        fig = plt.figure()
        plt.imshow(images[1], cmap='gray')
        plt.axis('off')
        plt.savefig('./imagestore/image{:04d}.png'.format(epoch))

    return GAN    




## Grayscale image
def rgb2gray(rgb):
    return tf.tensordot(rgb[...,:3], [0.5, 0.5, 0.5], axes = 1)