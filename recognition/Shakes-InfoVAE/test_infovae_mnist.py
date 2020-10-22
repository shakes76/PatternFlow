'''
MNIST InfoVAE with MMD

@author Shakes
'''
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.models import Model
import layers 

print('TensorFlow version:', tf.__version__)

#parameters
epochs = 2 #multiple of 3 + 1
batch_size = 64
depth = 32
kernel = 3
latent_size = 2

#load data
print("> Loading images ...")
# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, y_test) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
#x_train = (x_train - 127.5) / 127.5 # Normalize the images to [-1, 1]
#x_test = (x_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))
total_training, xSize, ySize, c = x_train.shape
print(x_train.shape)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(total_training).batch(batch_size)

#Build networks
#discriminator network
def encoder_network(input_shape, z_dim, name='E'):
    '''
    Encodes images into latent space
    '''
    input=Input(input_shape, name=name+'input')
    net=layers.Norm_Conv2D(input, depth, kernel_size=kernel, strides=2, activation=LeakyReLU(alpha=0.1)) #downsample
    net=layers.Norm_Conv2D(net, depth*2, kernel_size=kernel, strides=2, activation=LeakyReLU(alpha=0.1)) #downsample
    dense=Flatten()(net)
    dense=Dense(1024, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    dense=Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal())(dense)
    latent=Dense(z_dim, kernel_initializer=GlorotNormal())(dense)
    return Model(inputs=input, outputs=latent, name=name)

def decoder_network(input_shape, activation, name='D'):
    '''
    Decodes latent space into images
    '''
    input=Input(input_shape, name=name+'input')
    dense=Dense(128, activation=ReLU(), kernel_initializer=GlorotNormal())(input)
    dense=Dense(1024, activation=ReLU(), kernel_initializer=GlorotNormal())(dense)
    dense=Dense(7*7*depth*2, activation=ReLU(), kernel_initializer=GlorotNormal())(dense)
    dense=Reshape((7, 7, depth*2))(dense)
    net=layers.Norm_Conv2DTranspose(dense, depth*2, kernel_size=kernel, strides=2, activation=ReLU()) #upsample
    net=layers.Norm_Conv2DTranspose(net, depth, kernel_size=kernel, strides=2, activation=ReLU()) #upsample
    network=layers.Norm_Conv2D(net, 1, kernel_size=[1,1], strides=1, activation=activation) #RF 7 with stride
    return Model(inputs=input, outputs=network, name=name)

#Build models


#train


#test


#plot

print('END')
