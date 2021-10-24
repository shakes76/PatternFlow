import matplotlib.pyplot as plt
import numpy as np
import modules.dataset as dataset
from PIL import Image

def show_subplot(original, reconstructed):
    """
    Plot the original image against the reconstructed image
    
    Params:
        original(array): the original image to plot
        reconstructed(array): the reconstructed image get from the vqvae model
    """
    #plot the original image
    plt.figure(figsize = (10,12))
    plt.subplot(1,2,1);
    plt.imshow(original, cmap = 'gray')
    plt.title("Original")
    plt.axis("off")
    
    #plot the reconstructed image
    plt.subplot(1,2,2);
    plt.imshow(reconstructed, cmap = 'gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

def plot_images(img_count, data_test, vqvae_trainer):
    """
    Randomly select img_count number of images in the test dataset, test it on the model and
    plot the output(reconstruction image) against the original image
    
    Params:
        img_count: the number of samples to test
        data_test: the test dataset
        vqvae_trainer: the trained vqvaue model
    """
    #select img_count images randomly from the test dataset
    idx = np.random.choice(len(data_test), img_count)
    for i in range (img_count):
        original_img = Image.open(data_test[idx[i]])
        original_img = np.asarray(original_img)
        img = dataset.preprocess_image(original_img)
        img = np.expand_dims(img, axis=0) 
        #test the image on the model
        reconstruction_img = vqvae_trainer.predict(img)
        reconstruction_img = reconstruction_img * 255
        #plot
        show_subplot(original_img, reconstruction_img[0])