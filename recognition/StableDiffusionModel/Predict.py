import numpy as np
import tensorflow.keras as kr
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import *
from dataset import *


# PRECALCULATED VALUES / CONSTANTS
TIME_STEPS = 200
BETA = np.linspace(0.0001, 0.02, TIME_STEPS)
ALPHA = 1 - BETA
ALPHA_HAT = np.cumprod(ALPHA, 0)
ALPHA_HAT = np.concatenate((np.array([1.]), ALPHA_HAT[:-1]), axis=0)
SQRT_ALPHA_HAT = np.sqrt(ALPHA_HAT)
SQRT_ALPHA_HAT_COMPLIMENT = np.sqrt(1-ALPHA_HAT)


def denoise(input, predictedNoise, t):
    """ Denoises the input image at timestep t, given the predicted noise """
    alphaAtTime = np.take(ALPHA, t)
    alphaHatAtTime = np.take(ALPHA_HAT, t)

    coefficient = (1 - alphaAtTime) / (1 - alphaHatAtTime) ** .5
    mean = (1 / (alphaAtTime ** .5)) * (input - coefficient * predictedNoise)

    var = np.take(BETA, t)
    z = np.random.normal(size=input.shape)

    return mean + (var ** .5) * z

def showDenoiseProcess(model, decoder) :
    """
    Runs the denosing process 
    on a sample of pure noise:
    """

    # Creating pure noise to denoise
    x = tf.random.normal((1,32,32,1))
    imageList = []
    imageList.append(np.squeeze(np.squeeze(x, 0),-1))

    col = 0
    fig, axs = plt.subplots(2,5)

    for i in range(TIME_STEPS):
        t = np.expand_dims(np.array(TIME_STEPS-i-1, np.int32), 0)
        predictedNoise = model(x, t)

        # Performs denoising step
        x = denoise(x, predictedNoise, t)
        imageList.append(np.squeeze(np.squeeze(x, 0),-1))

        
        if (i % 50) == 0:
            nextImage = np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint8)

            # Plotting to subplot
            axs[0,col].imshow(tf.squeeze(nextImage), cmap="Greys")
            axs[0,col].set_title("t = {i}".format(i=i))
            axs[0,col].get_xaxis().set_visible(False)
            axs[0,col].get_yaxis().set_visible(False)

            axs[1,col].imshow(tf.squeeze(decoder(nextImage[None, :, :, :])), cmap="Greys")
            axs[1,col].set_title("t = {i}".format(i=i))
            axs[1,col].get_xaxis().set_visible(False)
            axs[1,col].get_yaxis().set_visible(False)

            col += 1

        axs[0,col].imshow(tf.squeeze(nextImage), cmap="Greys")
        axs[0,col].set_title("FINAL LATENT")
        axs[0,col].get_xaxis().set_visible(False)
        axs[0,col].get_yaxis().set_visible(False)


        axs[1,col].imshow(tf.squeeze(decoder(nextImage[None, :, :, :])), cmap="Greys")
        axs[1,col].set_title("FINAL IMAGE")
        axs[1,col].get_xaxis().set_visible(False)
        axs[1,col].get_yaxis().set_visible(False)
    nextImage = np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint8)

    plt.show()


if __name__ ==  "__main__":
    """ Main function """
    
    # Loading the AutoEncoder
    autoEnc = AutoEncoder(128, 32, kr.activations.relu, normLayers = True)
    autoEnc.load_weights("./Weights/FinalAutoEncoder")
    
    # Loading the diffusion Model
    unet = Unet()
    unet.load_weights("./Weights/FinalDiffusionModel")

    # Extracting the decoder
    decoder = autoEnc.buildDecoder(32)

    showDenoiseProcess(unet, decoder)


    
    