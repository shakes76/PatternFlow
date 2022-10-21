'''
OASIS Brains DCGAN Implementation

@author Peter Ngo

7/11/2020
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import numpy as np
import os
import time
import layers
import losses

# Check Tensorflow version
print("Tensorflow version " + tf.__version__)

#parameters
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 1200
IMG_SIZE = [256, 256]
CUR_DIR = os.path.dirname(os.path.abspath(__file__)) #absolute path to driver script.
PATH = os.path.join(CUR_DIR, "keras_png_slices_data/keras_png_slices_train/") #directory for training images.
gen_images_path = os.path.join(CUR_DIR, "generated_images/") #directory for predictions.
os.makedirs(gen_images_path, exist_ok=True) #do not overwrite existing.
plots_path = os.path.join(CUR_DIR, "plots/") #directory for plots.
os.makedirs(plots_path, exist_ok=True)

# Load the dataset
print('Loading the OASIS Brain dataset.....')
image_file_paths = tf.io.gfile.glob(PATH + '*')
#Set the buffer to be larger than the size of images.
BUFFER_SIZE = len(image_file_paths) + 1
#shuffle the image file paths.
image_file_paths = tf.data.Dataset.from_tensor_slices(image_file_paths).shuffle(BUFFER_SIZE)

# Load and preprocess the images.
def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels = 0) #keep original color channels.
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 127.5) / 127.5 # pixel values set -1 to 1

    return tf.image.resize(image, IMG_SIZE)

#map over each filepath to training images and parse and batch the images.
training_ds = image_file_paths.map(parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

#check batch size and image shape.
image_batch = next(iter(training_ds))
no_inputs, input_width, input_height, input_channels = image_batch.shape
print('Finished loading the OASIS Brain dataset.')


#model parameters
noise_dimensions = 100 #length of noise vector
gen_input_shape = (noise_dimensions, ) #generator input shape
disc_input_shape = (input_width, input_height, input_channels) #discriminator input shape

# Build the networks
def generator_network(input_shape):
    '''
    Receive random noise with gaussian distribution.
    Outputs image.
    '''
    input = Input(shape=input_shape)

    dense = layers.FullyConnected(input, 8*8*512, reshape_shape=(8,8,512))
    
    fconv1 = layers.Generator_Norm_Conv2DTranspose(dense, filters=512)
    fconv2 = layers.Generator_Norm_Conv2DTranspose(fconv1, filters=256)
    fconv3 = layers.Generator_Norm_Conv2DTranspose(fconv2, filters=128)
    fconv4 = layers.Generator_Norm_Conv2DTranspose(fconv3, filters=128)
    fconv5 = layers.Generator_Tanh_Conv2DTranspose(fconv4, filters=1) #output a 256x256 Tensor with 1 Channel
   
    return Model(inputs=input, outputs=fconv5, name="generator")

def discriminator_network(input_shape):
    '''
    Receive generator output image and real images from dataset.
    Outputs binary classficiation as a scalar.
    '''
    input = Input(shape=input_shape)
    conv1 = layers.Discriminator_Norm_Dropout_Conv2D(input, filters=64, dropout=0.15)
    conv2 = layers.Discriminator_Norm_Dropout_Conv2D(conv1, filters=128, dropout=0.15)
    conv3 = layers.Discriminator_Norm_Dropout_Conv2D(conv2, filters=256, dropout=0.15)
    conv4 = layers.Discriminator_Norm_Dropout_Conv2D(conv3, filters=512, dropout=0.1)
    output = layers.Flatten_Dense(conv4) #output a decision as a scalar

    return Model(inputs=input, outputs=output, name="discriminator")


#build generator
generator = generator_network(gen_input_shape)
generator.summary()

#build discriminator
discriminator = discriminator_network(disc_input_shape)
discriminator.summary()

#generator loss
def gen_loss(generated_outputs):
    return losses.generator_crossentropy(generated_outputs)

#discriminator loss
def disc_loss(generated_outputs, real_outputs):
    return losses.discriminator_crossentropy(generated_outputs, real_outputs)

#discriminator fake detection accuracy with threshold > 0.5
def disc_acc(generated_outputs, real_outputs):
    return losses.discriminator_accuracy(generated_outputs, real_outputs)

#optimizers
gen_opt = Adam(learning_rate = 0.0002)
disc_opt = Adam(learning_rate = 0.0001)

#prediction parameters
no_gen_images = 9 #number of generated brains to create
seed = tf.random.normal([no_gen_images, noise_dimensions]) # track progress of generated brains.


#Training
def train_step_dcgan(image_batch):
    #create 64 BATCH_SIZE latent variables sampled from a Gaussian with 100 Dimensions = noise dimensions.
    noise = tf.random.normal([BATCH_SIZE, noise_dimensions])
    #keep tabs on the gradients of the generator and discriminator.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_outputs = discriminator(image_batch, training=True)
        generated_outputs = discriminator(generated_images, training=True)

        generator_loss = gen_loss(generated_outputs) 
        discriminator_loss = disc_loss(generated_outputs, real_outputs)
        generated_acc, real_acc = disc_acc(generated_outputs, real_outputs)
    
    #differentiate the loss fn for generator and discriminator.
    gradients_of_gen = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables) 
    #update the trainable variables with the respective gradients.
    gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

    return generator_loss, discriminator_loss, generated_acc, real_acc

def train(dataset, epochs):
    # keep track of network losses and accuracy
    gen_losses = []
    disc_losses = []
    generated_accuracy = []
    real_accuracy = []
    print("Training........")
    for epoch in range(1, epochs+1):
        start = time.time()

        for image_batch in dataset:
            g_loss, d_loss, g_acc, r_acc = train_step_dcgan(image_batch)

        
        
        print('Time for epoch {} (disc_loss {}, gen_loss {}) is {} sec'.format(epoch, 
                                                                               d_loss, 
                                                                               g_loss, 
                                                                               time.time()-start))
        print('Discriminator accuracy on fake images {}, real images {}'.format(g_acc, r_acc))
        # Keep track of generator loss and discriminator accuracy and loss.                               
        gen_losses.append(g_loss)
        disc_losses.append(d_loss)
        generated_accuracy.append(g_acc)
        real_accuracy.append(r_acc)
        
        if (epoch % 10 == 0):
            # Produce images for the GIF every 10 epochs
            generate_and_save_images(generator, epoch, seed)

    return gen_losses, disc_losses, generated_accuracy, real_accuracy
        
#make predictions with the generator.
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(12, 12))

    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    #save the respective set of 9 predictions for later reference.
    plt.savefig(gen_images_path + 'epoch_{:04d}.png'.format(epoch), transparent=True)
    
#begin training
all_losses = train(training_ds, EPOCHS)

#plot the cross entropy loss of discriminator and generator.
plt.plot(np.arange(1, EPOCHS+1), all_losses[0], label = 'gen_loss')
plt.plot(np.arange(1, EPOCHS+1), all_losses[1], label = 'disc_loss')
plt.xlabel('epochs')
plt.ylabel('Cross Entropy Loss')
plt.title('DCGAN Cross Entropy Loss')
plt.legend()
plt.savefig(plots_path+'Cross_Entropy.png')
plt.show() #clear

#plot the discriminator classification accuracy on fake and real images.
plt.plot(np.arange(1,EPOCHS+1), all_losses[2], label = 'acc_fake_images')
plt.plot(np.arange(1,EPOCHS+1), all_losses[3], label = 'acc_real_images')
plt.xlabel('epochs')
plt.ylabel('Classification Accuracy')
plt.title('Discriminator Classification Accuracy')
plt.legend()
plt.savefig(plots_path+'Discriminator_Accuracy.png')
plt.show()

print('Finish')
