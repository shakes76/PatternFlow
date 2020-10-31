import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import math


class VAE(tf.keras.Model):
    def __init__(self, latent_dimension, kernel_size=3, strides=2):
        super(VAE, self).__init__()
        # number of dimensions of the latent distribution
        self.latent_dim = latent_dimension
        # the encoder 
        self.encoder = self.define_encoder(latent_dimension, kernel_size, strides)
        # the decoder
        self.decoder = self.define_decoder(latent_dimension, kernel_size, strides)

    @tf.function
    # sample a point from the latent distribution and decode it
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, sigmoid=True)

    def encode(self, x):
        parameters = self.encoder(x)
        mean, log_var = tf.split(parameters, num_or_size_splits=2, axis=1)
        return mean, log_var

    # implement the reparameterize trick
    def reparameterize(self, mean, log_var):
        # generate epsilon from a standard normal distribution
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * .5) + mean

    def decode(self, z, sigmoid=False):
        logits = self.decoder(z)
        if sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    # generate images by encoding and decoding the test samples
    def generate_images(self, test_sample):
        for x_test in test_sample:
            # encode the test samples and get the parameters of the latent distribution
            mean, log_var = model.encode(tf.expand_dims(x_test, axis=-1))
            # reparameter trick
            z = model.reparameterize(mean, log_var)
            # decode the points sampled from the latent distribution to generate images
            predictions = model.sample(z)
            return predictions

    # defining the encoder
    def define_encoder(self, latent_dimension, kernel_size, strides):
        e = Sequential()
        e.add(InputLayer(input_shape=(256, 256, 1)))
        e.add(Conv2D(filters=16, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization())
        e.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization())
        e.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization())
        e.add(Flatten())
        e.add(Dense(latent_dimension * 2))
        return e

    # defining the decoder
    def define_decoder(self, latent_dimension, kernel_size, strides):
        d = Sequential()
        d.add(InputLayer(input_shape=(latent_dimension,)))
        d.add(Dense(units=32 * 32 * 32, activation=tf.nn.relu))
        d.add(BatchNormalization())
        d.add(Reshape(target_shape=(32, 32, 32)))
        d.add(Conv2DTranspose(filters=64, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        d.add(Conv2DTranspose(filters=32, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        d.add(Conv2DTranspose(filters=16, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        d.add(Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=1, padding='same'))
        return d


# Calculate the similarity between the original test images and the generated images.
def calculate_ssim(predictions, test_sample):
    ssim_total = 0
    size = predictions.shape[0]
    for x_test in test_sample:
        for i in range(size):
            generated_img = img_as_float(tf.squeeze(predictions[i]))
            reference_img = img_as_float((tf.squeeze(x_test[i])))
            ssim_total += ssim(reference_img, generated_img, data_range=generated_img.max() - generated_img.min())
        # return the average structural similarity
        return ssim_total / size


def log_normal_pdf(sample, mean, log_var, raxis=1):
    log2pi = tf.math.log(2. * math.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi), axis=raxis)


def compute_loss(model, x):
    # get the parameters (mean and var) from the latent posterior distribution P(z|x)
    mean, log_var = model.encode(x)
    # genereate z from the latent distribution P(z|x) through the reparameter trick
    z = model.reparameterize(mean, log_var)
    # generate x through decoding z
    x_logit = model.decode(z)
    # calculate the cross entropy between the decoded x and the real x
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, log_var)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        # compute loss
        loss = compute_loss(model, x)
        # apply gradient descents to search for the optima
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# load the training and testing image datasets
def get_dataset(train_dir, test_dir, test_size=32):
    # a normalisation layer
    normalization_layer = Rescaling(1./255)
    # load the training images with the default batch size(i.e., 32)
    train_dataset = image_dataset_from_directory(train_dir, color_mode='grayscale', label_mode=None)
    # load the testing images with a specified batch size
    test_dataset = image_dataset_from_directory(test_dir, color_mode='grayscale', label_mode=None, batch_size=test_size)
    # normalise the training images
    normalized_train = train_dataset.map(lambda x: (normalization_layer(x)))
    # normalise the testing images
    normalized_test = test_dataset.map(lambda x: (normalization_layer(x)))
    # return the training and testing datasets
    return normalized_train, normalized_test


# plot the generated images 
def display_result(predictions):
    fig = plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = predictions[i]
        plt.imshow(tf.squeeze(img), cmap='gray')
        plt.axis('off')
    plt.show()


# function to train the model
def train(model, train_dataset, test_sample, epochs, optimizer):
    # iterate over all epochs
    for epoch in range(0, epochs + 1):
        # iterate over train_dataset containing training images
        for x_train in train_dataset:
            train_step(model, x_train, optimizer)
        # feed the network test samples to generate new images
        predictions = model.generate_images(test_sample)
        # display the results
        display_result(predictions)
        # evaluate the model using Structural Similarity between generated images and test samples
        print("> " + str(epoch) + ": SSIM = " + str(calculate_ssim(predictions, test_sample)))
    # return the trained model
    return model


# load pre-trained models
def load_pretrained_model(latent_dimension, encoder_name, decoder_name):
    # initialize a new VAE model
    model = VAE(latent_dimension)
    # load encoder
    model.encoder = load_model(encoder_name)
    # load decoder
    model.decoder = load_model(decoder_name)
    # return the loaded model
    return model


# save the trained models
def save_model(model, encoder_name, decoder_name):
    # save the encoder
    model.encoder.save(encoder_name)
    # save the decoder
    model.decoder.save(decoder_name)


if __name__ == '__main__':
    # define constants
    epochs = 10
    latent_dimension = 2
    train_img_dir = 'D:/keras_png_slices_data/keras_png_slices_data/directory'
    test_img_dir = 'D:/keras_png_slices_data/keras_png_slices_data/test directory'

    # use an Adam optimiser
    optimizer = Adam(1e-4)
    # load training and test datasets
    train_dataset, test_dataset = get_dataset(train_img_dir, test_img_dir, test_size=64)

    # initialize a new VAE model
    model = VAE(latent_dimension)
    # train a new model
    print('Start training')
    model = train(model, train_dataset, test_dataset, epochs, optimizer)
    save_model(model, 'encoder.h5', 'decoder.h5')