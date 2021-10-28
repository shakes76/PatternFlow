#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
import imageio


# In[2]:


#First loading all images into numpy arrays
#Note that this also normalises the data

def load_data():
    X_train=[]
    for f in listdir('train'):
        if f!='.ipynb_checkpoints':
            path='train/'+f
            X_train.append(list(imageio.imread(path)))
    X_train=np.array(X_train)

    X_test=[]
    for f in listdir('test'):
        path='test/'+f
        X_test.append(list(imageio.imread(path)))
    X_test=np.array(X_test)

    X_train = X_train[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]

    X_train = X_train.astype(float)/255.
    X_test = X_test.astype(float)/255.
    
    return X_train, X_test

X_train, X_test = load_data()

#Sanity check
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)


# In[3]:


class VQ(layers.Layer):

#N and K are respectively the number of vectors in the codebook the dimension of the vectors
    def __init__(self,N,K,beta=0.1,**kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.K = K
        self.beta = beta

        #As per the paper, initialising the prior as uniform

        prior = tf.random_uniform_initializer()
        self.codebook = tf.Variable(initial_value=prior((self.K,self.N),dtype="float32"),
                                    trainable=True,name="Codebook")
    
    def call(self,input):

        #Flattening the inputs
        input_shape = tf.shape(input)
        reshape_input = tf.reshape(input,[-1,self.K])

        #Moving on to the quantisation part
        idx = self.get_code_indices(reshape_input)
        #One-hot encoding the inputs first, so as to make it easier to compute.
        #Indeed this well help resort to a simple matrix product later on
        oh_enc = tf.one_hot(idx,self.N)
        #Then quantising and reshaping to the original input shape
        q_enc = tf.reshape(tf.matmul(oh_enc,self.codebook,transpose_b=True),input_shape)

        #Finally computing the loss
        #The below is the third loss term introduced in the paper, the one where beta intervenes
        beta_loss = self.beta*tf.reduce_mean((tf.stop_gradient(q_enc) - input)**2)
        #This one on the other hand is the second term, the one that measures the difference
        #between the encoded vector and the input
        encoding_loss = tf.reduce_mean((q_enc - tf.stop_gradient(input))**2)
        self.add_loss(beta_loss + encoding_loss)

        return input+tf.stop_gradient(q_enc-input)

    def get_code_indices(self,input):
        #Caveat : input needs to be flattened here
        #Here determining the nearest codebook vector to each of the input data points
        sim = tf.matmul(input,self.codebook)
        distances = tf.reduce_sum(input**2,axis=1,keepdims=True)+tf.reduce_sum(self.codebook**2,axis=0) - 2*sim
        #Lastly, need to chuck everything into an argmin to get indices of the codebook vectors each input data point was assigned to
        encoding_indices = tf.argmin(distances,axis=1)
        return encoding_indices

#Now we need to code the encoder and decoder
def encoder(dim=128):
    enc_inputs = keras.Input(shape=(256,256,1))

    x = layers.Conv2D(256, (3,3), activation="relu", strides=1, padding="same")(enc_inputs)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation="relu", strides=1, padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    x = layers.Dropout(0.25,seed=55)(x)
    x = layers.Conv2D(64, (3,3), activation="relu", strides=1, padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    x = layers.Dropout(0.125,seed=55)(x)
    enc_outputs = layers.Conv2D(dim,(3,3),padding='same')(x)

    return keras.Model(enc_inputs,enc_outputs,name='encoder')

def decoder(dim=128):
    latents = keras.Input(shape=encoder().output.shape[1:])
    x = layers.Dropout(0.125,seed=55)(latents)
    x = layers.Conv2DTranspose(64,(3,3),activation='relu',strides=(2,2),padding='same')(x)
    x = layers.Dropout(0.25,seed=55)(x)
    x = layers.Conv2DTranspose(128,(3,3),activation='relu',strides=(2,2),padding='same')(x)
    x = layers.Dropout(0.5,seed=55)(x)
    x = layers.Conv2DTranspose(256,(3,3),activation='relu',strides=(2,2),padding='same')(x)
    decoded = layers.Conv2DTranspose(1,3,padding='same')(x)
    return keras.Model(latents,decoded,name='decoder')

#Assembling all of the above to build the VQVAE
def vqvae(dim=128,N=512):

    vq = VQ(N,dim,name='Quantiser')
    enc = encoder(dim)
    dec = decoder(dim)
  
    input = keras.Input(shape=(256,256,1))
    output = enc(input)
    ql = vq(output)
  
    reconstructed = dec(ql)
    return keras.Model(input,reconstructed,name='VQVAE')

vqvae().summary()


# In[4]:


#On to the training module

class vqvae_trainer(keras.models.Model):

    def __init__(self,var,dim=128,N=512,**kwargs):
        super(vqvae_trainer,self).__init__(**kwargs)
        self.var = var
        self.dim = dim
        self.N = N
        self.vq_vae = vqvae(self.dim,self.N)

        #Tracking the losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name='vq_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker,self.reconstruction_loss_tracker,self.vq_loss_tracker]

    def train_step(self,input):
        with tf.GradientTape() as gt:
            reconstructions = self.vq_vae(input)
            reconstruction_loss = (tf.reduce_mean((input - reconstructions) ** 2) / self.var)
            total_loss = reconstruction_loss + sum(self.vq_vae.losses)
    
        #Time to backprop
        gradients = gt.gradient(total_loss,self.vq_vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.vq_vae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vq_vae.losses))

        return {"Total_loss": self.total_loss_tracker.result(),
              "reconstruction_loss": self.reconstruction_loss_tracker.result(),
              "VQVAE loss": self.vq_loss_tracker.result(),}


# In[5]:


train_var = np.var(X_train)
VQVAE_trainer = vqvae_trainer(train_var,dim=128,N=512)
VQVAE_trainer.compile(optimizer=keras.optimizers.Adam())


# In[6]:


#VQVAE_trainer.fit(X_train,epochs=20,batch_size=32)
VQVAE_trainer.load_weights('VQVAE')


# In[7]:


#Taking a look at reconstruction results
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(),cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze(),cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

trained_vqvae = VQVAE_trainer.vq_vae
idx = np.random.choice(len(X_test), 10)
test_images = X_test[idx]
reconstructions_test = trained_vqvae.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    show_subplot(test_image, reconstructed_image)


# In[8]:


#Let us now have a look at the codes

my_encoder = VQVAE_trainer.vq_vae.get_layer("encoder")
my_quantiser = VQVAE_trainer.vq_vae.get_layer("Quantiser")

encoded_outputs = my_encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = my_quantiser.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i].squeeze(),cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i],cmap='gray')
    plt.title("Code")
    plt.axis("off")
    plt.show()


# In[9]:


#Let's try to generate images with the codes using a PixelCNN

#Defining the hyperparameters first
n_residual_blocks = 2
n_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]
print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")


# In[10]:


class PixelCNNLayer(layers.Layer):
    
    def __init__(self,mask_type,**kwargs):
        super(PixelCNNLayer,self).__init__()
        self.mask_type = mask_type
        self.conv_layer = layers.Conv2D(**kwargs)
        
    def build(self,input_shape):
        self.conv_layer.build(input_shape)
        kernel_shape = self.conv_layer.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[:kernel_shape[0]//2,...] = 1.0
        self.mask[kernel_shape[0]//2,:kernel_shape[1]//2,...] = 1.0
        if self.mask_type == 'B':
            self.mask[kernel_shape[0]//2,kernel_shape[1]//2,...] = 1.0
        
    def call(self,inputs):
        self.conv_layer.kernel.assign(self.conv_layer.kernel*self.mask)
        return self.conv_layer(inputs)
    
class ResidualBlock(layers.Layer):
    
    def __init__(self,filters,**kwargs):
        super(ResidualBlock,self).__init__(**kwargs)
        self.conv_layer_1 = layers.Conv2D(filters=filters,kernel_size=1,activation='relu')
        self.pixel_conv_layer = PixelCNNLayer(mask_type='B',
                                             filters=filters//2,
                                             kernel_size=3,
                                             activation='relu',
                                             padding='same')
        self.conv_layer_2 = layers.Conv2D(filters=filters,kernel_size=1,activation='relu')
        
    def call(self,inputs):
        x=self.conv_layer_1(inputs)
        x=self.pixel_conv_layer(x)
        x=self.conv_layer_2(x)
        return layers.add([inputs,x])
    
PixelCNN_inputs = keras.Input(pixelcnn_input_shape,dtype=tf.int32)

one_hot = tf.one_hot(PixelCNN_inputs,VQVAE_trainer.N)

x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(one_hot)
x = PixelCNNLayer(mask_type='A',filters=512,kernel_size=32,activation='relu',padding='same')(x)
x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
for elt in range(n_residual_blocks):
    x = ResidualBlock(filters=512)(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
for elt in range(n_pixelcnn_layers):
    x = PixelCNNLayer(mask_type='B',filters=512,kernel_size=1,strides=1,
                     activation='relu',padding='valid')(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)

output_layer = layers.Conv2D(filters=VQVAE_trainer.N,kernel_size=1,strides=1,padding='same')(x)

PixelCNN = keras.Model(PixelCNN_inputs,output_layer,name='PixelCNN')
PixelCNN.summary()


# In[19]:


#Generating the codebook indices.
#print(encoded_outputs.shape)
#print(encoded_outputs.reshape(-1, encoded_outputs.shape[-1]).shape)
#print(X_train.shape)
for i in range(X_train.shape[0]):
    encoded_outputs = my_encoder.predict(X_train[i])
    print(encoded_outputs.reshape(-1, encoded_outputs.shape[-1]).shape)
    flat_enc_outputs = encoded_outputs.reshape(-1,encoded_outputs.shape[-1])
    codebook_indices = my_quantiser.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
#encoded_outputs = my_encoder.predict(X_train)
#flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
#codebook_indices = my_quantiser.get_code_indices(flat_enc_outputs)

#codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
#print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")


# In[ ]:




