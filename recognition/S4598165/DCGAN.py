#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tqdm
import numpy as np
from numpy import asarray, load, zeros, ones
from numpy.random import randn, randint
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, ReLU
from matplotlib import pyplot
import tensorflow as tf
from numpy import asarray
from os import listdir
from PIL import Image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        tf.config.experimental
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


# In[2]:


def define_discriminator(in_shape=(256,256,3)):
    model = Sequential()
    model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 16 * 16
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16,16, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    return model


# In[3]:


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def load_real_samples():
    data = load('brain_train.npz')
    X = data['arr_0']
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X

def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y


# In[5]:


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

# create a plot of generated images
def plot_generated(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :])
    pyplot.show()

def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels


# In[7]:


def Cal_SSIM(directory,img):
    required_size=(256, 256)
    max_ssim = 0
    ssim_list=[]
    k = 0
    
    for filename in listdir(directory):
        k+=1
        if k == 500:
            break
        pixels = load_image(directory + filename)
        image = Image.fromarray(pixels)
        image = image.resize(required_size)
        brain = asarray(image)
        brain= tf.convert_to_tensor(brain)
        img= tf.convert_to_tensor(img)
        brain = tf.cast(brain, dtype= tf.float32)
        res=tf.image.ssim(img, brain, 255)
        res=res.numpy()
        ssim_list.append(res)
        max_ssim=max(res, max_ssim)

    return max_ssim


# In[6]:


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('\r',end='',flush=True)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f ' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss),end='',flush=True)
        if (i+1) % 2 == 0:
            # generate images
            latent_points = generate_latent_points(100, 100)
            # generate images
            X  =g_model.predict(latent_points)
            # scale from [-1,1] to [0,1]
            X = (X + 1) / 2.0*255
            n=100
            res_ssim=[]
            Max_ssim = 0

            for k in tqdm.tqdm(range(n)):
                img=X[k, :, :]
                img1=img.copy()
                
                img = np.array(img,np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                ret3,th3 = cv2.threshold(img,52,255,0)
                contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    if area < 150:
                        cv2.drawContours(th3, [contours[i]], 0, 0, -1)
                for i in range(img1.shape[0]):
                    for j in range(img1.shape[1]):
                        if i<20or i>img1.shape[0]-20:
                            img1[i][j][0]= 0
                            img1[i][j][1]= 0
                            img1[i][j][2]= 0

                        if j<20or j>img1.shape[1]-20:
                            img1[i][j][0] = 0
                            img1[i][j][1]= 0
                            img1[i][j][2]= 0

                        if th3[i][j]==0:
                            img1[i][j][0]=0
                            img1[i][j][1]= 0
                            img1[i][j][2]= 0
                
                res=Cal_SSIM('../keras_png_slices_data/keras_png_slices_test/',img1)

                res_ssim.append(res)
            if Max_ssim<sum(res_ssim)/n:
                Max_ssim=sum(res_ssim)/n
                print(">%d,SSIM = " %(i+1,Max_ssim))
                filename = 'generator_model_%.3f.h5' % (Max_ssim)
                g_model.save(filename)
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# In[1]:


latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
n_epochs = 40
n_batch = 16
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch)

