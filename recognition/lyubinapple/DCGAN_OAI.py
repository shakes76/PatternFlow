'''
    File name: DCGAN_OAI.py
    Author: Bin Lyu(45740165)
    Date created: 10/30/2020
    Date last modified: 
    Python Version: 4.7.4
'''
import tensorflow as tf
from os import listdir
from numpy import asarray, load, zeros, ones, savez_compressed
from numpy.random import randn, randint
from PIL import Image
from matplotlib import pyplot
import glob
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import Reshape, LeakyReLU, BatchNormalization
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_images(path, n_images):
    images = list()
    # enumerate files
    for fn in listdir(path):
        # load the image
        image = Image.open(path + fn)
        image = image.convert('RGB')
        image = image.resize((80, 80),Image.ANTIALIAS)
        pixels = asarray(image)
        # save image into list
        images.append(pixels)
        # stop once we have enough
        if len(images) >= n_images:
            break
    return asarray(images)

def plot_images(imgs, n):
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(imgs[i])
    pyplot.show()

path = "/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/AKOA_Analysis/"
images = load_images(path, 150)
savez_compressed('OAI.npz', images)
print('Loaded:', images.shape)
plot_images(images, 5)

# create discriminator model, a binary classification CNN model
def define_discriminator(ishape=(80,80,3)):
    model = Sequential()
    # normal
    model.add(Conv2D(128, (5,5), padding='same', input_shape=ishape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization()
    # downsample to 40x40
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # use binary classification activation function 
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # compile model, adam stochastic gradient descent
    # learning rate=0.0002, momentum=0.5
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

    # create generator model, generate 80×80 color image from a point in latent space
def define_generator(latent_size):
    model = Sequential()
    # foundation for 5x5 feature maps
    n_nodes = 128 * 5 * 5
    # use fully connected layer to interpret the point in latent space
    # gan tries to map the input distribution in latent space to generate new output
    model.add(Dense(n_nodes, input_dim=latent_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    # upsample to 10x10
    # use transpose convolutional layer to increase the area of activations to 4 times
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 80x80
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 80x80x3, scale to [-1,1]
    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    # no need to compile due to this model is not trained directly
    model.summary()
    return model

    # define the combined generator and discriminator model
# train the generator model weight 
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    # ensure only train and update the model weights in the generator
    d_model.trainable = False
    
    model = Sequential()
    # combine generator and discriminator model together
    model.add(g_model)
    model.add(d_model)
    model.summary()
    
    # compile gan model, adam stochastic gradient descent
    # learning rate=0.0002, momentum=0.5
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

    # load dataset that have pre-processed as np array and rescale
def load_real_samples():
    data = load('OAI.npz')
    X = data['arr_0']
    X = X.astype('float32')
    # normalization - scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

    # select real samples, select random images every batch
def generate_real_samples(dataset, n_samples):
    # randomly choose a number and retrieve the image
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # these are real images, so class labels = 1
    y = ones((n_samples, 1))
    return X, y

    # create input for the generator
def create_latent_points(latent_size, n_samples):
    # generate points in the latent space (random normal distribution)
    x_input = randn(latent_size * n_samples)
    # reshape into a batch of inputs for the generator model
    x_input = x_input.reshape(n_samples, latent_size)
    return x_input

    # use generator model to generate fake examples, with class labels
def generate_fake_samples(g_model, latent_size, n_samples):
    # generate points in latent space as a batch of input samples for g_model
    x_input = create_latent_points(latent_size, n_samples)
    # predict outputs, the fake images
    X = g_model.predict(x_input)
    # these are fake images, so class labels = 0
    y = zeros((n_samples, 1))
    return X, y

# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot(n rows, n cols, index)
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    fn = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(fn)
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_size, n_samples=100):
    # prepare real images and labels(label=1)
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate the discriminator on real examples, just need acc_real
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake images and labels(label=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_size, n_samples)
    # evaluate the discriminator on fake examples, just need acc_fake
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize the discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    # save plot and the generator model tile file
    save_plot(x_fake, epoch)
    fn = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(fn)

    # train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_size, ep=50, batch_num=128):
    # each epoch includes about 390 (50000/128) batches
    bat_per_epo = int(dataset.shape[0] / batch_num)
    half_batch = int(batch_num / 2)
    # manually enumerate epochs
    for i in range(ep):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples on half batch
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            # run a single gradient update of a batch of samples
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples on half batch
            X_fake, y_fake = generate_fake_samples(g_model, latent_size, half_batch)
            # update discriminator model weights
            # run a single gradient update of a batch of samples
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = create_latent_points(latent_size, batch_num)
            # create inverted labels for the fake samples
            # so the generator can provide a larger gradient during training
            # updating the generator toward getting better at generating real samples on the next batch
            y_gan = ones((batch_num, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # call the summarize_performance() function every 10 training epochs
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_size)

# size of the latent space
latent_size = 100

# create discriminator model
print("create discriminator model ")
d_model = define_discriminator()

# create generator model
g_model = define_generator(latent_size)

# create gan model (combination of the above two model)
gan_model = define_gan(g_model, d_model)

# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_size)

#Compute SSIM similarity
pred_img = tf.convert_to_tensor(X)
true_img = load_images(folder_images, 25)
true_img = tf.convert_to_tensor(true_img)
pred_i = tf.image.convert_image_dtype(pred_img, tf.float32)
true_i = tf.image.convert_image_dtype(true_img, tf.float32)
ssim2 = tf.image.ssim(pred_i, true_i, max_val=1)
final_result = tf.reduce_mean(tf.cast(ssim2, dtype=tf.float32))
print("SSIM is: ", final_result)