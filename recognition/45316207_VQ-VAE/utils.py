"""
utils.py

Alex Nicholson (45316207)
11/10/2022

Contains extra utility functions to help with things like plotting visualisations and ssim calculation

"""


from re import A
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras


def show_reconstruction_examples(model, test_data, num_to_show):
    """
    Shows a series of tiled plots with side-by-side examples of the original data and reconstructed data

        Parameters:
            model (Keras Model): VQ VAE Model
            test_data (ndarray): Test dataset of real brain MRI images
            num_to_show (int): Number of reconstruction comparison examples to show
    """

    # Visualise output generations from the finished model
    idx = np.random.choice(len(test_data), num_to_show)

    test_images = test_data[idx]
    reconstructions_test = model.predict(test_images)

    for i in range(reconstructions_test.shape[0]):
        original = test_images[i, :, :, 0]
        reconstructed = reconstructions_test[i, :, :, 0]
        
        plt.figure()
        
        plt.subplot(1, 2, 1)
        plt.imshow(original.squeeze() + 0.5, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed.squeeze() + 0.5, cmap='gray')
        plt.title("Reconstructed (ssim: {:.2f})".format(get_image_ssim(original, reconstructed)))
        plt.axis("off")

        plt.savefig('out/original_vs_reconstructed_{:04d}.png'.format(i))
        plt.close()

def l2_dist_scratch(x, y):
    # Can do at least 2500 images
    # Pretty sure works for 5000 and maybe 7000
    sx = np.sum(x**2, axis=1, keepdims=True)
    sy = np.sum(y**2, axis=1, keepdims=True)
    distances = np.sqrt(-2 * x.dot(y.T) + sx + sy.T)
    return distances


def get_code_indices_savedmodel(vector_quantizer, flattened_inputs):
    """
    Gets the indices of the codebook vectors???

        Parameters:
            (Tensorflow Tensor): purpose???

        Returns:
            encoding_indices (Tensorflow Tensor): purpose???
    """

    # Calculate L2-normalized distance between the inputs and the codes.
    distances = l2_dist_scratch(flattened_inputs, vector_quantizer.embeddings.numpy().T)

    # Derive the indices for minimum distances.
    encoding_indices = tf.argmin(distances, axis=1)

    return encoding_indices.numpy()






def plot_training_metrics(history):
    """
    Shows a series of tiled plots with side-by-side examples of the original data and reconstructed data

        Parameters:
            history (???): The training history (list of metrics over time) for the model
    """
    num_epochs = len(history.history["loss"])

    # Plot losses
    plt.figure()
    plt.plot(range(1, num_epochs+1), history.history["loss"], label='Total Loss', marker='o')
    plt.plot(range(1, num_epochs+1), history.history["reconstruction_loss"], label='Reconstruction Loss', marker='o')
    plt.plot(range(1, num_epochs+1), history.history["vqvae_loss"], label='VQ VAE Loss', marker='o')
    plt.title('Training Losses', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.xticks(range(1, num_epochs+1))
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('out/training_loss_curves.png')
    plt.close()

    # Plot log losses
    plt.figure()
    plt.plot(range(1, num_epochs+1), history.history["loss"], label='Log Total Loss', marker='o')
    plt.plot(range(1, num_epochs+1), history.history["reconstruction_loss"], label='Log Reconstruction Loss', marker='o')
    plt.plot(range(1, num_epochs+1), history.history["vqvae_loss"], label='Log VQ VAE Loss', marker='o')
    plt.title('Training Log Losses', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.xticks(range(1, num_epochs+1))
    plt.ylabel('Log Loss', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('out/training_logloss_curves.png')
    plt.close()

def plot_ssim_history(ssim_history):
    """
    Shows a series of tiled plots with side-by-side examples of the original data and reconstructed data

        Parameters:
            history (???): The training history (list of metrics over time) for the model
    """
    num_epochs = len(ssim_history)

    # SSIM History
    plt.figure()
    plt.plot(range(1, num_epochs+1), ssim_history, label='Average Model SSIM', marker='o')
    plt.title('Model SSIM Performance Over Time', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.xticks(range(1, num_epochs+1))
    plt.ylabel('Average Model SSIM', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('out/training_ssim_curve.png')
    plt.close()


def get_image_ssim(image1, image2):
    """
    Gets the ssim between 2 images

        Parameters:
            image1 (ndarray): An image
            image2 (ndarray): A second image to compare with the first one

        Returns:
            ssim (int): The structural similarity index between the two given images
    """
    similarity = ssim(image1, image2,
                  data_range=image1.max() - image1.min())

    return similarity


def get_model_ssim(model, test_data):
    """
    Gets the average ssim of a model

        Parameters:
            model (ndarray): The VQ VAE model
            test_data (ndarray): Test dataset of real brain MRI images

        Returns:
            ssim (int): The  average structural similarity index achieved by the model
    """

    sample_size = 10 # The number of generations to average over

    similarity_scores = []

    # Visualise output generations from the finished model
    idx = np.random.choice(len(test_data), 10)

    test_images = test_data[idx]
    reconstructions_test = model.predict(test_images)

    for i in range(reconstructions_test.shape[0]):
        original = test_images[i, :, :, 0]
        reconstructed = reconstructions_test[i, :, :, 0]

        similarity_scores.append(ssim(original, reconstructed, data_range=original.max() - original.min()))

    average_similarity = np.average(similarity_scores)

    return average_similarity









# TODO: Document this function
def visualise_codebook_sampling(vqvae_model, pixelcnn_model, train_data, num_embeddings, examples_to_show):
    # Create a mini sampler model.
    inputs = tf.keras.layers.Input(shape=pixelcnn_model.input_shape[1:])
    outputs = pixelcnn_model(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)


    # Construct a prior to generate images
    # Create an empty array of priors
    priors = np.zeros(shape=(examples_to_show,) + (pixelcnn_model.input_shape)[1:])
    examples_to_show, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel
    print(f"Generating priors... (this step may take a long time)")
    for row in range(rows): # 64
        print(f"Row {row}/{rows}")
        for col in range(cols): # 64
            # Feed the whole array and retrieving the pixel value probabilities for the next pixel
            probs = sampler.predict(priors, verbose=0)
            # Use the probabilities to pick pixel values and append the values to the priors
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")

    # Now use the decoder to generate the images
    # Perform an embedding lookup.
    pretrained_embeddings = vqvae_model.get_layer("vector_quantizer").embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    encoder_output_shape = vqvae_model.get_layer("encoder").predict(train_data).shape[1:]
    quantized = tf.reshape(quantized, (-1, *(encoder_output_shape)))

    # Generate novel images.
    decoder = vqvae_model.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    for i in range(examples_to_show):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig('out/novel_generation_{:04d}.png'.format(i))
        plt.close


# TODO: Document this function
def visualise_codes(model, test_data, num_to_show):
    print("#########################")
    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")
    print(quantizer)
    print(type(quantizer))
    print("#########################")

    idx = np.random.choice(len(test_data), num_to_show)
    test_images = test_data[idx]

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = get_code_indices_savedmodel(quantizer, flat_enc_outputs)
    codebook_indices = codebook_indices.reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i])
        plt.title("Code")
        plt.axis("off")
        plt.savefig('out/code_{:04d}.png'.format(i))
        plt.close