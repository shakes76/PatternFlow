import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import load_model
import VAE
from matplotlib import pyplot as plt
import sys


# Calculate the similarity between the original test images and the generated images.
def calculate_ssim(predictions, test_sample):
    ssim_total = 0
    size = predictions.shape[0]
    # summing up the ssim of different pairs of images and work out the average ssim
    for x_test in test_sample:
        for i in range(size):
            # the generated image
            generated_img = tf.image.convert_image_dtype(predictions[i], dtype=tf.float32)
            # the reference image
            reference_img = tf.image.convert_image_dtype(x_test[i], dtype=tf.float32)
            # the ssim of this pair of images
            ssim_total += tf.image.ssim(reference_img, generated_img, max_val=1.0)
        # return the average structural similarity
        return ssim_total / size


# compute the log probability
def log_normal_pdf(sample, mean, log_var, ax=1):
    pi = 3.1415926536
    log_2pi = tf.math.log(2.0 * pi)
    p = -0.5 * ((sample - mean) ** 2.0 * tf.exp(-log_var) + log_var + log_2pi)
    return tf.reduce_sum(p, axis=ax)


def calculate_loss(model, x):
    # get the parameters (mean and var) from the latent posterior distribution P(z|x)
    z_mean, z_log_var = model.encode(x)
    # genereate z from the latent distribution P(z|x) through the reparameter trick
    z = model.reparameterize(z_mean, z_log_var)
    # generate x through decoding z
    predictions = model.decode(z)
    # reconstruction loss
    reconst_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=x)
    # likelihood P(x|z)
    px_z = -tf.reduce_sum(reconst_loss, axis=[1, 2, 3])
    # prior P(z) is modelled by a unit Gaussian distribution.
    # var = 1 => log_var = 0
    pz = log_normal_pdf(z, 0., 0.)
    # posterior distribution q(z|x)
    qz_x = log_normal_pdf(z, z_mean, z_log_var)
    # For simplicity we compute P(z) - P(z|x), which is a regularizer,
    # measuring the difference between the two distributions.
    return -tf.reduce_mean(px_z + pz - qz_x)


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        # compute loss
        loss = calculate_loss(model, x)
        # apply gradient descents to search for the optima
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# load the training and testing image datasets
def get_dataset(train_dir, test_dir, test_size=32):
    # a normalisation layer
    normalization_layer = Rescaling(1. / 255)
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
    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        img = predictions[i]
        plt.imshow(tf.squeeze(img), cmap='gray')
        plt.axis('off')
    plt.show()


# function to train the model
def train(model, train_dataset, test_dataset, epochs, optimizer):
    # statistics to store
    elbos = []
    ssims = []
    print('Starting training...')
    # iterate over all epochs
    for epoch in range(0, epochs + 1):
        # iterate over train_dataset containing training images
        for x_train in train_dataset:
            train_step(model, x_train, optimizer)
        # feed the network test samples to generate new images
        predictions = model.generate_images(model, test_dataset)

        # display the results
        try:
            display_result(predictions)
        except:
            pass

        loss = Mean()
        for test_x in test_dataset:
            loss(calculate_loss(model, test_x))
        elbo = -loss.result()
        # evaluate the model using Structural Similarity between generated images and test samples and ELBO
        ssim = calculate_ssim(predictions, test_dataset)
        print("> " + str(epoch) + ": SSIM=" + str(ssim) + ', ELBO=' + str(elbo))
        # add the evaluatons to a list and plot the results later
        ssims.append(ssim)
        elbos.append(elbo)
    # return the trained model
    return model, elbos, ssims


# load pre-trained models
def load_pretrained_model(latent_dimension, encoder_name, decoder_name):
    # initialize a new VAE model
    model = VAE.VAENetwork(latent_dimension)
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
    epochs = 30
    latent_dimension = 2

    # train_img_dir stores the training images
    train_img_dir = sys.argv[1]
    # test_img_dir stores the testing images
    test_img_dir = sys.argv[2]

    # use an Adam optimiser
    optimizer = Adam(1e-4)
    # load training and test datasets
    train_dataset, test_dataset = get_dataset(train_img_dir, test_img_dir, test_size=554)

    # initialize a new VAE model
    model = VAE.VAENetwork(latent_dimension)
    # train a new model
    model, elbos, ssims = train(model, train_dataset, test_dataset, epochs, optimizer)
    # save the trained models
    try:
        save_model(model, 'encoder.h5', 'decoder.h5')
    except:
        pass

    # plot the ssim and elbo.
    try:
        xs = range(0, epochs + 1)
        fig = plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        plt.title('SSIM of Each Epoch')
        plt.plot(xs, ssims)
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.subplot(1, 2, 2)
        plt.title('ELBO of Each Epoch')
        plt.plot(xs, elbos, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        fig.tight_layout()
    except:
        print("matplotlib not supported")

