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
    Randomly select img_count number of images from the test dataset, test it on the vqvae model and
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
        #generate the reconstructed image using the vqvae model
        reconstruction_img = vqvae_trainer.predict(img)
        reconstruction_img = reconstruction_img * 255
        #plot
        show_subplot(original_img, reconstruction_img[0])

def calc_mean(x, pred):
    """
    Calculate the mean pixel value for two images respectively
    
    Params:
        x: the original image
        pred: the reconstructed image
        
    Returns:
        The mean pixel value for the orignal image and the reconstructed image respectively
    """
    luminance_x = 0
    luminance_pred = 0
    pixels = 256*256

    for row in range(256):
        for col in range(256):
            luminance_x += x[row][col]
            luminance_pred += pred[row][col]
    
    luminance_x = luminance_x / pixels
    luminance_pred = luminance_pred /pixels
    return luminance_x, luminance_pred

def calc_std(x, pred, mean_x, mean_pred):
    """
    Calculate the pixel standard deviation value for two images respectively
    
    Params:
        x: the original image
        pred: the reconstructed image
        mean_x: the mean pixel value for the original image
        mean_pred: the mean pixel value for the reconstrcuted image
        
    Returns:
        The pixel standard deviation value for the orignal image and the reconstructed image respectively
    """
    var_x = 0
    var_pred = 0
    pixels = 256*256-1

    for row in range(256):
        for col in range(256):
            var_x += np.square(x[row][col] - mean_x)
            var_pred += np.square(pred[row][col] - mean_pred)
    
    var_x = np.sqrt(var_x/pixels)
    var_pred = np.sqrt(var_pred/pixels)
    return var_x, var_pred

def calc_covariance(x, pred, mean_x, mean_pred):
    """
    Calculate the covranice value of two images
    
    Params:
        x: the original image
        pred: the reconstructed image
        mean_x: the mean pixel value for the original image
        mean_pred: the mean pixel value for the reconstrcuted image
        
    Returns:
        The pixel covariance value for the orignal image and the reconstructed image
    """
    covar = 0
    pixels = 256*256-1

    for row in range(256):
        for col in range(256):
            covar += (x[row][col] - mean_x)*(pred[row][col] - mean_pred)
    
    return covar/pixels

def ssim(mean_x, mean_pred, std_x, std_pred, covariance):
    """
    Calculate the structured similarity between two images
    
    Params:
        mean_x: the mean pixel value for the original image
        mean_pred: the mean pixel value for the reconstrcuted image
        std_x: the piexel standard deviation for the original image
        std_pred: the pixel standard deviation for the reconstrcuted image
        covariance: the pixel covariance value for the orignal image and the reconstructed image
    
    Returns:
        The structured similarity of the original and reconstructed images
    """
    #k1 = 0.01 and k2 = 0.03 by default
    #L is the dynamic range of the pixel-values(2^bits per pixel -1)
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = np.square(K1*L)
    C2 = np.square(K2*L)
    C3 = C2/2
    
    l_x_y = (2*mean_x*mean_pred + C1)/(np.square(mean_x)+np.square(mean_pred)+C1)
    c_x_y = (2*std_x*std_pred + C2)/(np.square(std_x)+np.square(std_pred)+C2)
    s_x_y = (covariance+C3)/(std_x+std_pred+C3)
    return l_x_y * c_x_y * s_x_y

def mean_ssim(data_test, vqvae_trainer):
    """
    Calculate the mean structured similiarity on the whole test dataset
    
    Params:
        data_test: the test dataset
        vqvae_trainer: the trained vqvaue model
    
    Returns:
        the mean structured similiarity over the whole test dataset
    """
    ssim_coef = 0
    for i in range(len(data_test)):
        #for each test image, preprocess it
        original_img = Image.open(data_test[i])
        original_img = np.asarray(original_img)
        img = dataset.preprocess_image(original_img)
        img = np.expand_dims(img, axis=0) 
        #put the processed image into the vqvae model to get the reconstructed image
        reconstruction_img = vqvae_trainer.predict(img)
        img = img[0,:,:,0]
        reconstruction_img = reconstruction_img[0,:,:,0]

        #calculate the structured similiarity between the original image and the reconstructed image
        #and add it to the total ssim
        mean_x, mean_pred = calc_mean(img,reconstruction_img)
        std_x, std_pred = calc_std(img,reconstruction_img, mean_x, mean_pred)
        covariance = calc_covariance(img,reconstruction_img, mean_x, mean_pred)
        ssim_coef += ssim(mean_x, mean_pred, std_x, std_pred, covariance)
    #return the mean ssim coefficient
    return ssim_coef/len(data_test)

def get_cnn_shape(encoder, data_test):
    """
    Get the output shape of the vqvae encoder 
    
    Params:
        encoder: the vqvae encorder
        data_test: the test dataset
    
    Returns:
        The output shape of the vqvae encoder
    """
    #open the first image in the testing dataset
    original_img = Image.open(data_test[0])
    original_img = np.asarray(original_img)
    img = dataset.preprocess_image(original_img)
    img = np.expand_dims(img, axis=0) 
    #predict the image on the encoder
    encoded_outputs = encoder.predict(img)
    return encoded_outputs.shape