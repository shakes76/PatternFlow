import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from model.modules import latent_dims, get_indices, pixelcnn_input_shape, image_shape, encoded_image_shape

def visualize_autoencoder(model, data, n):
    """
    visualize_autoencoder
    args: model - the autoencoder to visualize
          data  - data to visualize from
          n     - number of samples from data at random
    plots the original, encoded form, and decoded form side by side
    """
    data = tf.random.shuffle(data)
    predictions = model.predict(tf.reshape(data[0:n], shape=(-1,*image_shape[0:2])))
    encoded = get_indices(model.vq.variables[0], tf.reshape(model.encoder.predict(data), shape=(-1,latent_dims)), quantize=False, splits=32)
    encoded = tf.reshape(encoded, shape=(-1, *pixelcnn_input_shape[0:2]))
    plt.tight_layout()
    fig, axs = plt.subplots(n, 3, figsize=image_shape[0:2])
    for i in range(n):
        axs[i,0].imshow(tf.reshape(data[i], shape=image_shape[0:2]))
        axs[i,2].imshow(predictions[i])
        axs[i,1].imshow(tf.reshape(encoded[i], shape=pixelcnn_input_shape))
    plt.show()

def calculate_ssim(model, data):
    """
    calculate_ssim
    args: model - vqvae to check ssim for
          data  - data to check ssim for
    prints average SSIM and percent over 0.6 for the data and model provided
    """
    test_data = tf.reshape(data, shape=(-1,*image_shape[0:2]))

    results = model.predict(test_data)

    ssim = tf.image.ssim(tf.expand_dims(test_data, axis=-1), results, max_val=1.0)

    mean, pct = (tf.reduce_mean(ssim).numpy(), tf.math.reduce_sum(tf.cast(tf.math.greater(ssim,0.6), dtype=tf.float64)).numpy() / len(data))

    print("Average SSIM:", mean, " Percent >0.6:", pct)

def visualize_pixelcnn(pixelcnn, vqvae, n):
    """
    visualize_pixelcnn
    args: pixelcnn - pixelcnn to generate encoded images
          vqvae    - vqvae used to decode images
          n        - number of images to generate
    plots the original, encoded form, and decoded form side by side
    """
    n = 8
    generated = tf.zeros(shape=(n,*pixelcnn_input_shape[0:2]))
    generated = tf.reshape(generated, shape=(n, *pixelcnn_input_shape[0:2]))
    generated = generated.numpy()
    for row in range(pixelcnn_input_shape[0]):
        for col in range(pixelcnn_input_shape[1]):
            probabilities = pixelcnn.predict(generated)[:, row, col]

            generated[:, row, col] = tf.reshape(
                tf.random.categorical(probabilities, 1), shape=(n,))

    plt.close()
    plt.tight_layout()
    fig, axs = plt.subplots(n, 2, figsize=image_shape[0:2])
    for i in range(n):
        axs[i, 0].imshow(generated[i])
        pred = vqvae.decoder.predict(tf.reshape(tf.gather(vqvae.vq.variables[0], tf.cast(
            generated[i], dtype=tf.int64)), shape=(1, *encoded_image_shape)))
        axs[i, 1].imshow(tf.reshape(pred, shape=image_shape[0:2]))
    plt.show()
