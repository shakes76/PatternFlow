import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from vqvae import VQVAE, get_closest_embedding_indices

import numpy as np
import matplotlib.pyplot as plt

def show_image_and_reconstruction(original, cb, reconstructed):
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cb)
    plt.title("Codebook")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    IMG_SIZE = 256

    # load OASIS images from folder
    dataset = image_dataset_from_directory("keras_png_slices_data/keras_png_slices_train", 
                                          label_mode=None, 
                                          image_size=(IMG_SIZE, IMG_SIZE),
                                          color_mode="grayscale",
                                          batch_size=32)

    # normalize pixels between [-0.5, 0.5]
    dataset = dataset.map(lambda x: (x / 255.0) - 0.5)

    # calculate variance of training data (at a individual pixel level) to pass into VQVAE
    count = dataset.unbatch().reduce(tf.cast(0, tf.int64), lambda x,_: x + 1 ).numpy()
    mean = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + y ).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE)
    var = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + tf.math.pow(y - mean,2)).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE - 1)

    # hyperparameters which seem to give the best results so far
    learning_rate = 2e-4
    beta = 1.5
    latent_dim = 2
    num_embeddings = 512
    epochs = 30
    batch_size = 32

    input_size = (IMG_SIZE, IMG_SIZE, 1)

    # create model
    vqvae_model = VQVAE(input_size, latent_dim, num_embeddings, beta, var)
    vqvae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # fit it
    vqvae_model.fit(dataset, epochs=epochs, batch_size=batch_size)

    # visualise some reconstructions

    num_batches_to_show = 2
    num_images_per_batch_to_show = 5

    test_images_batches = dataset.take(num_batches_to_show)
    for test_images in test_images_batches.as_numpy_iterator():
        reconstructions = vqvae_model.predict(test_images)

        encoder_outputs = vqvae_model.encoder().predict(test_images)
        encoder_outputs_flat = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])

        codebook_indices = get_closest_embedding_indices(vqvae_model.quantizer().embeddings(), encoder_outputs_flat)
        codebook_indices = codebook_indices.numpy().reshape(encoder_outputs.shape[:-1])

        for i in range(num_images_per_batch_to_show):
            # add 0.5 back to the images to undo the initial scaling (i.e. go from [-0.5, 0.5] to [0,1])
            original_image = tf.reshape(test_images[i], (1, IMG_SIZE, IMG_SIZE, 1)) + 0.5
            reconstructed_image = tf.reshape(reconstructions[i], (1, IMG_SIZE, IMG_SIZE, 1)) + 0.5
            codebook_image = codebook_indices[i]

            show_image_and_reconstruction(tf.squeeze(original_image), codebook_image, tf.squeeze(reconstructed_image))
            ssim = tf.math.reduce_sum(tf.image.ssim(original_image, reconstructed_image, max_val=1.0)).numpy()
            print("SSIM: ", ssim)
