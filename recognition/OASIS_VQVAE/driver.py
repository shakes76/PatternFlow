import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from vqvae import VQVAE, get_closest_embedding_indices
from pixelcnn import PixelCNN

import numpy as np
import matplotlib.pyplot as plt

import argparse

IMG_SIZE = 256

def calculate_training_variance(dataset):
    """
    Calculates the pixel level variance over the dataset
    """
    count = dataset.unbatch().reduce(tf.cast(0, tf.int64), lambda x,_: x + 1 ).numpy()
    mean = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + y ).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE)
    var = dataset.unbatch().reduce(tf.cast(0, tf.float32), lambda x,y: x + tf.math.pow(y - mean,2)).numpy().flatten().sum() / (count * IMG_SIZE * IMG_SIZE - 1)
    return var


class SSIMCallback(tf.keras.callbacks.Callback):
    """
    Custom metric callback for calculating SSIMs
    """
    def __init__(self, validation_data, shift=0.0):
        super(SSIMCallback, self).__init__()
        self._val = validation_data
        self._shift = shift

    def on_epoch_end(self, epoch, logs):
        total_count = 0.0
        total_ssim = 0.0

        for batch in self._val:
            recon = self.model.predict(batch)
            total_ssim += tf.math.reduce_sum(tf.image.ssim(batch + self._shift, recon + self._shift, max_val=1.0))
            total_count += batch.shape[0]

        logs['val_avg_ssim'] = (total_ssim/total_count).numpy()
        print("epoch: {:d} - val_avg_ssim: {:.6f}".format(epoch, logs['val_avg_ssim']))


def plot_history(history):
    """
    Plots the loss histories
    """
    plt.plot(history.history['loss'])
    plt.title('total training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['reconstruction_loss'])
    plt.title('training reconstruction loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['vq_loss'])
    plt.title('training quantizer loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['val_avg_ssim'])
    plt.title('validation dataset average SSIM')
    plt.ylabel('average SSIM')
    plt.xlabel('epoch')
    plt.show()


def show_reconstruction_examples(model, images, shift):
    """
    Shows (original_image, codebook, reconstruction) for the list of images
    """
    reconstructions = model.predict(images)

    encoder_outputs = model.encoder().predict(images)
    encoder_outputs_flat = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])

    codebook_indices = get_closest_embedding_indices(model.quantizer().embeddings(), encoder_outputs_flat)
    codebook_indices = codebook_indices.numpy().reshape(encoder_outputs.shape[:-1])

    for i in range(len(images)):
        # add the shift back to the images to undo the initial shifting (e.g. go from [-0.5, 0.5] to [0,1])
        original_image = tf.reshape(images[i], (1, IMG_SIZE, IMG_SIZE, 1)) + shift
        reconstructed_image = tf.reshape(reconstructions[i], (1, IMG_SIZE, IMG_SIZE, 1)) + shift
        codebook_image = codebook_indices[i]

        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(original_image), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(codebook_image)
        plt.title("Codebook")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(reconstructed_image), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        plt.show()
        ssim = tf.math.reduce_sum(tf.image.ssim(original_image, reconstructed_image, max_val=1.0)).numpy()
        print("SSIM: ", ssim)


def load_images(location, image_size, batch_size):
    """
    Create dataset from OASIS images in given location
    """
    return image_dataset_from_directory(location, 
                                        label_mode=None, 
                                        image_size=image_size,
                                        color_mode="grayscale",
                                        batch_size=batch_size,
                                        shuffle=True)


def get_codebook_mapper_fn(encoder, embeddings):
    """
    Returns a mapper function handle that can be passed to the dataset.map function.
    This function encodes the images into codebook indices
    """
    def mapper(x):
        encoded_outputs = encoder(x)
        flat_enc_outputs = tf.reshape(encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])
        codebook_indices = get_closest_embedding_indices(embeddings, flat_enc_outputs)
        codebook_indices = tf.reshape(codebook_indices, tf.shape(encoded_outputs)[:-1])
        return codebook_indices

    return mapper


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="VQVAE trainer")

    parser.add_argument("--data", 
        type=str, 
        default="keras_png_slices_data", 
        help="Location of unzipped OASIS dataset. Folder should contain the folders 'keras_png_slices_train' and 'keras_png_slices_validate'.")

    parser.add_argument("--K", "--num-embeddings", type=int, default=512, help="Number of embeddings, described as K in the VQVAE paper (default: 512)")
    parser.add_argument("--D", "--embedding-dim", type=int, default=2, help="Size of the embedding vectors, described as D in the VQVAE paper (default: 2)")
    parser.add_argument("--beta", type=float, default=1.5, help="Committment cost, described as beta in VQVAE paper (default: 1.5)")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for ADAM optimiser for the VQVAE trainer (default 2e-4)")

    parser.add_argument("--shift", type=float, default=0.5, help="Normalisation shift that will be subtracted from each pixel value prior to training (default: 0.5)")

    ## PixelCNN args
    parser.add_argument("--pcnn-epochs", type=float, default=10, help="Number of epochs to train the PixelCNN (default: 10)")
    parser.add_argument("--pcnn-learning-rate", type=float, default=3e-4, help="Learning rate for ADAM optimiser for the PixelCNN trainer (default: 3e-4)")
    parser.add_argument("--pcnn-filters", type=int, default=128, help="Number of filters to use in PixelCNN (default: 128)")
    parser.add_argument("--pcnn-res-blocks", type=int, default=2, help="Number of residual blocks to use in PixelCNN (default: 2)")
    parser.add_argument("--pcnn-layers", type=int, default=2, help="Number of extra convolutional layers to use in PixelCNN (default: 2)")

    args = parser.parse_args()

    # load OASIS images from folder
    dataset             = load_images(args.data + "/keras_png_slices_train", (IMG_SIZE, IMG_SIZE), args.batch_size)
    dataset_validation  = load_images(args.data + "/keras_png_slices_validate", (IMG_SIZE, IMG_SIZE), args.batch_size)

    # normalize pixels (in [0,255]) between [-SHIFT, -SHIFT + 1] (for example: [-0.5, 0.5])
    dataset             = dataset.map(lambda x: (x / 255.0) - args.shift)
    dataset_validation  = dataset_validation.map(lambda x: (x / 255.0) - args.shift)

    # calculate variance of training data (at a individual pixel level) to pass into VQVAE
    training_variance = calculate_training_variance(dataset)

    # create model
    input_size = (IMG_SIZE, IMG_SIZE, 1)
    vqvae_model = VQVAE(input_size, args.D, args.K, args.beta, training_variance)

    vqvae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

    # fit it
    history = vqvae_model.fit(dataset,
                              epochs=args.epochs, 
                              batch_size=args.batch_size, 
                              callbacks=[SSIMCallback(dataset_validation, args.shift)])

    # plot history
    plot_history(history)

    # plot some example images, codebooks, and reconstructions
    # Need to use this loop hack because of tf.Dataset limitations I can't find a workaround of yet
    for test_images in dataset.take(1).as_numpy_iterator():
        show_reconstruction_examples(vqvae_model, test_images[1:10], args.shift)

    ### PIXELCNN ###

    # Map dataset to their codebooks using the trained VQVAE
    codebook_mapper = get_codebook_mapper_fn(vqvae_model.encoder(), vqvae_model.quantizer().embeddings())
    codebook_dataset = dataset.map(codebook_mapper)

    # create PixelCNN and train it
    pcnn_input_shape = vqvae_model.encoder().output.shape[1:3]

    pixel_cnn = PixelCNN(pcnn_input_shape, args.K, args.pcnn_filters, args.pcnn_res_blocks, args.pcnn_layers)
    pixel_cnn.compile(optimizer=tf.keras.optimizers.Adam(args.pcnn_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    pixel_cnn.fit(codebook_dataset, batch_size=args.batch_size, epochs=args.pcnn_epochs)
