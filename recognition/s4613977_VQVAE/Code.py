from os import listdir
from os.path import isfile, join
import numpy as np
import imageio
import matplotlib.pyplot as plt

#First loading all images into numpy arrays
#Note that this also normalises the data

def load_data():
    X_train=[]
    for f in listdir('train'):
        path='train/'+f
        X_train.append(list(imageio.imread(path)))
    X_train=np.array(X_train)

    X_test=[]
    for f in listdir('test'):
        path='test/'+f
        X_test.append(list(imageio.imread(path)))
    X_test=np.array(X_test)

    X_val=[]
    for f in listdir('val'):
        path='val/'+f
        X_val.append(list(imageio.imread(path)))
    X_val=np.array(X_val)

    X_train = X_train[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]
    X_val = X_val[:,:,:,np.newaxis]

    X_train = X_train.astype(float)/255.
    X_test = X_test.astype(float)/255.
    X_val = X_val.astype(float)/255.
    
    return X_train, X_test, X_val

#Sanity check
X_train, X_test, X_val = load_data()

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("X_val shape:",X_val.shape)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

class VQ(layers.Layer):

  #N and K are respectively the number of vectors in the codebook the dimension of the vectors
  def __init__(self,N,K,beta=1,**kwargs):
    super().__init__(**kwargs)
    self.N = N
    self.K = K
    self.beta = beta

    #As per the paper, initialising the prior as uniform

    prior = tf.random_uniform_initializer()
    self.codebook = tf.Variable(initial_value=prior((self.K,self.N),dtype="float32"),
                                trainable=True,name="Codebooke")
    
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
    beta_loss = self.beta*tf.reduce_mean(tf.stop_gradient(q_enc - input)**2)
    #This one on the other hand is the second term, the one that measures the difference
    #between the encoded vector and the input
    encoding_loss = tf.reduce_mean((q_enc - tf.stop_gradient(input)**2))
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

  x = layers.Conv2D(512, (3,3), activation="relu", strides=1, padding="same")(enc_inputs)
  x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
  x = layers.Dropout(0.5,seed=55)(x)
  x = layers.Conv2D(256, (3,3), activation="relu", strides=1, padding="same")(x)
  x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
  x = layers.Dropout(0.25,seed=55)(x)
  x = layers.Conv2D(128, (3,3), activation="relu", strides=1, padding='same')(x)
  x = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
  x = layers.Dropout(0.125,seed=55)(x)
  enc_outputs = layers.Conv2D(dim,(1,1),padding='same')(x)

  return keras.Model(enc_inputs,enc_outputs,name='encoder')

def decoder(dim=128):
  input_shape = encoder().output.shape[1:]
  latents = keras.Input(shape=input_shape)
  x = layers.Dropout(0.125,seed=55)(latents)
  x = layers.Conv2DTranspose(128,(3,3),activation='relu',strides=1,padding='same')(x)
  x = layers.Dropout(0.25,seed=55)(x)
  x = layers.Conv2DTranspose(256,(3,3),activation='relu',strides=1,padding='same')(x)
  x = layers.Dropout(0.5,seed=55)(x)
  x = layers.Conv2DTranspose(512,(3,3),activation='relu',strides=1,padding='same')(x)
  return keras.Model(latents,x,name='decoder')

#Assembling all of the above to build the VQVAE
def vqvae(dim=128,N=512):

  vq = VQ(N,dim,name='Quantiser')
  enc = encoder(dim)
  dec = decoder(dim)
  
  input = keras.Input(shape=(256,256,1))
  output = enc(input)
  ql = vq(output)
  
  reconstructed_images = dec(ql)
  return keras.Model(input,reconstructed_images,name='VQVAE')

vqvae().summary()

#On to the training module

class vqvae_trainer(keras.models.Model):

  def __init__(self,var,dim=128,N=512,**kwargs):
    super(vqvae_trainer,self).__init__(**kwargs)
    self.var = var
    self.dim = dim
    self.N = N
    self.vq_vae = vqvae(self.dim,self.N)
    
    #Tracking the losses
    self.loss = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
    self.vq_loss = keras.metrics.Mean(name='VQ_loss')

    @property
    def metrics(self):
      return [self.loss,self.reconstruction_loss,self.vq_loss]

    def train_step(self,input):
      with tf.GradientTape() as gt:
        rec = self.vq_vae(input)
        rec_loss = (tf.reduce_mean((input - rec) ** 2) / self.var)
        loss = rec_loss + sum(self.vq_vae.losses)
      
      #Time to backprop
      gradients = gt.gradient(loss,self.vq_vae.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients,self.vq_vae.trainable_variables))

      self.loss.update_state(loss)
      self.reconstruction_loss.update_state(rec_loss)
      self.vq_loss.update_state(sum(self.vq_vae.losses))

      return {"Total loss": self.loss.result(),
            "reconstruction loss": self.reconstruction_loss.result(),
            "VQVAE loss": self.vq_loss.result(),}
      
      
