"""
Shows example of trained model
Prints out results and provides visualisations 
Written by Daniel Sayer 2022
For 21/10/22
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import models

from train import dice_similarity, train_model

def load_saved_model(filename = "improved_unet.model"):
    """
    Loads a saved file - saved as "improved_unet.model" (from train_model)
    Returns: saved model
    """

    return models.load_model(filename, custom_objects={"dice_similarity": dice_similarity})

def plot_metrics(model_fit, is_loss_plot):
    """
    Shows and saves a visual representation of the epoch training data
    Param: model_fit - The model in which the training is based on (model_fit = model.fit(params))
    Param: (bool) is_loss_plot: denotes whether to graph the loss of the training
        - True: plots the loss function
        - False: plots the dice similarity
    Returns: Saved image, either loss_plot.png or dice_similarity_plot.png depending on is_loss_plot
    """
    if is_loss_plot:
        type = 'loss'
    else: 
        type = 'dice_similarity'

    #Plot
    plt.plot(model_fit.history[type], label= f'Training {type}')
    plt.plot(model_fit.history[f'val_{type}'], label=f'Validation {type}')
    #Presentation
    plt.title(f'Test vs Validation {type}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(type)
    plt.savefig(f'{type}_plot.png')
    plt.show()
    return

def plot_predicted_masks(images, masks, model, number_of_samples):
    """
    Plots the images vs the mask vs the predicted mask
    Param: images - the image set in which to graph
    Param: masks - the masks set in which to graph
    Param: model - the training model in which the predictions are to be formed from 
        Note: can use model = load_model()
    Param: number_of_samples - the number of images to plot, will take n random images
        from the image data set
    Returns: saved image - "mask.png" with a visual representation of the predictions vs the given data
    """
    sample = random.sample(range(0, 1999), number_of_samples)
    predicted_masks = model.predict(images)

    mask_plot, axs = plt.subplots(number_of_samples, 3)

    #Labels
    axs[0,0].set_title("Original Image")  # type: ignore
    axs[0,1].set_title("Given Mask")  # type: ignore
    axs[0,2].set_title("Prediction")  # type: ignore
    for number, i in enumerate(sample):
        #Image
        axs[number, 0].imshow(images[i])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             # type: ignore
        #Mask
        axs[number, 1].imshow(tf.argmax(masks[i], axis = 2))  # type: ignore
        #Prediction
        axs[number, 2].imshow(tf.argmax(predicted_masks[i], axis = 2))  # type: ignore
        
        #Set all axes labels to null as images not graphs
        for a in [0, 1, 2]:
            axs[number, a].set_xticks([])  # type: ignore
            axs[number, a].set_yticks([])  # type: ignore


    mask_plot.savefig('mask.png')
    mask_plot.show()    