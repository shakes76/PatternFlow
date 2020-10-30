import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import math


class VAE(tf.keras.Model):
    def __init__(self, latent_dimsion, kernel_size=3, strides=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dimsion
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(256, 256, 1)),
                Conv2D(filters=16, kernel_size=kernel_size, strides=strides, activation='relu'),
                BatchNormalization(),
                Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation='relu'),
                BatchNormalization(),
                Conv2D(filters=64, kernel_size=kernel_size, strides=strides, activation='relu'),
                BatchNormalization(),
                Flatten(),
                # No activation
                Dense(latent_dimsion * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dimsion,)),
                Dense(units=32 * 32 * 32, activation=tf.nn.relu),
                BatchNormalization(),
                Reshape(target_shape=(32, 32, 32)),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                # No activation
                Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

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
        # encode the test samples and get the parameters of the latent distribution
        mean, log_var = model.encode(tf.expand_dims(test_sample, axis=-1))
        # reparameter trick
        z = model.reparameterize(mean, log_var)
        # decode the points sampled from the latent distribution to generate images
        predictions = model.sample(z)
        return predictions


# Calculate the similarity between the original test images and the generated images.
def calculate_ssim(predictions, test_sample):
    ssim_total = 0
    #
    size = predictions.shape[0]
    for i in range(size):
        generated_img = predictions[i, :, :, 0]
        reference_img = test_sample[i, :, :, 0]
        generated_img = img_as_float(generated_img)
        reference_img = img_as_float(reference_img)
        ssim_total += ssim(reference_img, generated_img, data_range=generated_img.max() - generated_img.min())
    # return the average structural similarity
    return ssim_total/size


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


def load_img_to_tensor(ds, crop_ratio):
    list_of_batches = list(ds.as_numpy_iterator())
    brains = list()
    for batch in list_of_batches:
        for images in batch:
            for i in range(images.shape[0]):
                if (len(images[i].shape) == 3):
                    image = tf.image.central_crop(images[i], crop_ratio)/255
                    brains.append(image)
    brain_images = tf.convert_to_tensor(brains, dtype=tf.float32)
    return brain_images


def get_dataset(train_dir, test_dir, batch_size, crop_ratio=1):
    # load the images to BatchDataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale')
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_dir, color_mode='grayscale')
    train_dataset = load_img_to_tensor(train_dataset, crop_ratio)
    test_dataset = load_img_to_tensor(test_dataset, crop_ratio)
    train_size = train_dataset.shape[0]
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(train_size).batch(batch_size))
    return train_dataset, test_dataset


# function to train the model
def train(model, train_dataset, test_sample, epochs, optimizer):
    # iterate over all epochs
    for epoch in range(0, epochs + 1):
        # iterate over train_dataset containing training images
        for x_train in train_dataset:
            train_step(model, x_train, optimizer)
        # feed the network test samples to generate new images
        predictions = model.generate_images(test_sample)
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
    batch_size = 32
    train_img_dir = 'D:/keras_png_slices_data/keras_png_slices_data/directory'
    test_img_dir = 'D:/keras_png_slices_data/keras_png_slices_data/test directory'

    # use an Adam optimiser
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # load training and test datasets
    train_dataset, test_dataset = get_dataset(train_img_dir, test_img_dir, batch_size)
    # load pre-trained model or train a new model
    load_or_train = input('Load existing model or train a new model? T/L:')
    if load_or_train == 'L':
        # load pre-trained model
        model = load_pretrained_model(latent_dimension, 'encoder669.h5', 'decoder669.h5')
        predictions = model.generate_images(test_dataset)
        print("> SSIM = " + str(calculate_ssim(predictions, test_dataset)))
        print(model.encoder.summary())
    else:
        # initialize a new VAE model
        model = VAE(latent_dimension)
        # train a new model
        print('Start training')
        model = train(model, train_dataset, test_dataset, epochs, optimizer)
        save_model(model, 'new_encoder.h5', 'new_decoder.h5')


