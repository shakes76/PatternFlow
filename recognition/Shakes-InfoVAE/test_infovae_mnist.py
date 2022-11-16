'''
MNIST InfoVAE with MMD

@author Shakes
'''
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
import layers 

print('TensorFlow version:', tf.__version__)

#parameters
epochs = 5 #multiple of 3 + 1
batch_size = 64
depth = 32
kernel = 3
latent_size = 2
activation = Activation('relu')
model_name = "InfoVAE-2D"
path = 'E:/Dev/PatternFlow/recognition/Shakes-InfoVAE/'

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

mse = tf.keras.losses.MeanSquaredError()

#Build models
z_size = (latent_size, )
input_shape = (xSize, ySize, 1)

#build encoder
encoder = encoder_network(input_shape, latent_size)
encoder.summary(line_length=133)

#build decoder
decoder = decoder_network(z_size, activation)
decoder.summary(line_length=133)

#losses
def encoder_loss(latent):
    '''
    Compute MMD loss for the InfoVAE
    '''
    def compute_kernel(x, y):
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
        tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

    'So, we first get the mmd loss'
    'First, sample from random noise'
    batch_size = K.shape(latent)[0]
    latent_dim = K.int_shape(latent)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    'calculate mmd loss'
    loss_mmd = compute_mmd(true_samples, latent)

    'Add them together, then you can get the final loss'
    return loss_mmd

def decoder_loss(y_true, y_pred):
    '''
    Returns reconstruction loss as L2
    '''
    return mse(y_true, y_pred)

#build VAE
input = Input(input_shape, name=model_name+'_input')
z = encoder(input)
recon = decoder(z)
vae = Model(inputs=input, outputs=recon, name=model_name)
vae.summary(line_length=133)

vae_opt = tf.keras.optimizers.Adam(1e-4)

#train functions
num_examples = 16
seed = tf.random.normal([num_examples, latent_size, 1])

@tf.function #compiles function, much faster
def train_step_vae(images):
    '''
    The training step with the gradient tape (persistent). The switch allows for for different training schedules.
    '''
    with tf.GradientTape() as vae_tape:
        latent_codes = encoder(images, training=True)
        recons = decoder(latent_codes, training=True)
        
        mmd_loss = encoder_loss(latent_codes)
        recon_loss = decoder_loss(images, recons)

        loss = mmd_loss + recon_loss

    gradients_of_vae = vae_tape.gradient(loss, vae.trainable_variables)
    vae_opt.apply_gradients(zip(gradients_of_vae, vae.trainable_variables))

    return loss

def train(dataset, epochs):
    losses = []
    for epoch in range(1, epochs+1):
        start = time.time()

        loss = -1
        for image_batch in dataset:
            loss = train_step_vae(image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(decoder, epoch, seed)
        
        print('Time for epoch {} (loss {}) is {} sec'.format(epoch, loss, time.time()-start))
        losses.append(loss)
    
    return losses

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(path+'image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    
def save_images(epoch, test_input):
    # Save images and image plot
    fig = plt.figure(figsize=(4,4))

    for i in range(test_input.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(test_input[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(path+'image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

#train
convergence = train(train_dataset, epochs)

#predict
#reconstruct test set
test_set = x_test[0:num_examples,...]
save_images(0, test_set)
generate_and_save_images(vae, -1, test_set)

#plot
plt.plot(np.arange(epochs), convergence)

print('END')
