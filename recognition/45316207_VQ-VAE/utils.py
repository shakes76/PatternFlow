"""
utils.py

Alex Nicholson (45316207)
11/10/2022

Contains extra utility functions to help with things like plotting visualisations and ssim calculation

"""


import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


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
    plt.savefig('training_loss_curves.png')
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
    plt.savefig('training_logloss_curves.png')
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
    plt.savefig('training_ssim_curve.png')
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