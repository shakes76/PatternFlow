from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil
import csv

from modules import StyleGAN


def denormalise(data: np.array , denormalisation_mean: float) -> np.array:
    """
    Take output of GAN and undo data preprocessing procedure to return
    a matrix of integer greyscale image pixel intensities suitable to
    convert directly into an image.

    Args:
        data (np.array): Normalised data
        denormalisation_mean (float): Mean used to center original data

    Returns:
        np.array: Denormalized data
    """
    data = np.array(data) #cast from any arraylike (Allows Tensor Compatibility)
    decentered = data + denormalisation_mean
    return (decentered * 255).astype(np.uint8)

def create_image(data: np.array, filename: str = None) -> Image:
    """
    Creates an new PNG image from a generated data matrix.
    Saves image to output folder if a name is specified

    Args:
        data (np.array): uint8 single channel matrix of greyscale image 
                pixel intensities
        name (str or NoneType, optional): Filename and path to save image. 
                Folder path must be valid if provided. 
                If None image is not saved. Defaults to None.

    Returns:
        Image: Generated image
    """
    #minor duplication tolerated to keep everything wrapped in a class/function
    TARGET_RES = 256
    PADDING = 35 
    
    
    im = Image.fromarray(data[:,:,0],'L').resize(
                (TARGET_RES-PADDING,TARGET_RES-PADDING) )

    #"decompress" to original size by upsampling then then padding back border
    #sample colour of top left pixel (background) so padding has matching colour
    back = Image.new('L', (TARGET_RES,TARGET_RES), im.getdata()[0])  
    back.paste(im,(PADDING//2,PADDING//2))
    im = back.convert("RGBA")
    if filename is not None:
        im.save(filename+".png")
    return im

def random_generator_inputs(
        num_images: int,
        latent_dim: int,
        noise_start: int,
        noise_end: int
        ) -> list[np.array]:
    """
    Helper function that generates a set of random inputs that acts as a 
    complete set of input for styleGAN generator

    Args:
        num_images (int): Number of inputs (and hence output images from 
                generator) in returned tensors 
        latent_dim (int): Latent dimension of styleGAN inputs are for
        noise_start (int): Side length of first noise input required for 
                styleGAN
        noise_end (int): Side length of last noise input required for styleGAN

    Returns:
        list[np.array]: List of tensors that can be passed in as input for 
        styleGAN generator
    """

    latent_vectors = tf.random.normal(shape = (num_images,latent_dim))
    input_tensors = [latent_vectors]
    curr_res = noise_start
    while curr_res <= noise_end:
        input_tensors.append(tf.random.normal(
                shape = (num_images,curr_res,curr_res,latent_dim)
                ))
        input_tensors.append(tf.random.normal(
                shape = (num_images,curr_res,curr_res,latent_dim)
                ))
        curr_res = curr_res*2
    return input_tensors

def make_fresh_folder(folder_path: str) -> None:
    """
    Generates a folder at the specified location, wiping any existing contents
    If the specified folder already existed. Parents are created if neccessary,
    but are not cleared if they exist already.

    Args:
        folder_path (str): Path to specified folder. If no folder exists at the 
                location a new one is created. If a folder exists at the 
                location it is wiped.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def save_training_history(history: dict[list[float]], filename: str) -> None: 
    """
    Appends training history of a training epoch to a specified csv file. 
    Creates a new file if the specified file does not exist. Collumns of output 
    file represent the tracked history of each passed metric. 
    Each row of the file is a given epoch

    Args:
        history (dict[list[float]]): Training history for a training epoch. 
                Each key-value pair should represent a metric name paired with 
                the list of ordered values representing the values of the metric 
                per epoch of training. 
        filename (str): Path to csv file in which history will be appended. 
                If a file extention is not included .npy will be appended.
    """
     
    with open(filename, mode = 'a', newline='') as f:
        #take average of batch losses for the given epoch for each metric
        csv.writer(f).writerow([
                sum(history[metric])/len(history[metric]) for metric in history
            ])

def load_training_history(csv_location: str) -> dict[list[float]]:
    """
    Loads training history from specified csv file.

    Args:
        csv_location (str): Path to csv file from which to load training history. 
                Precondition: Specified file should have the correct number of 
                collumns (len(StyleGAN.METRICS))

    Returns:
        dict[list[float]]: Training history. Each key-value represents a metric 
                name paired with the list of ordered values representing the 
                values of the metric per batch of training. 
    """
    history = {metric: [] for metric in StyleGAN.METRICS}
    with open(csv_location, mode= 'r', newline= '') as f:
        reader = csv.DictReader(f, fieldnames= StyleGAN.METRICS)
        for row in reader:
            for metric in StyleGAN.METRICS:
                history[metric].append(row[metric])
    return history 



def plot_training(
        history: dict[list[float]], 
        output_file: str, 
        epoch_range: tuple[int,int] = None
        ) -> None:
    """
    Generate a plot of the each training metric against epoch. The three lines 
    will be presented on a single figure for comparison.

    Args:
        history (dict[list[float]]): Training history. Each key-value pair 
                should represent a metric name paired with the list of ordered 
                values representing the values of the metric per epoch of 
                training.
        output_file (str): Path to which to save the generated figure. If the 
                specified file already exists it will be overwritten.
        epoch_range (tuple[int,int], optional): Set of epochs to plot between. 
                Set to None to plot entire training history. Defaults to None.
    """
    
    #truncate to specified range if required
    if epoch_range is not None:
        start,end = epoch_range
        history = {metric: 
                history[metric][start:end] 
                for metric in history
                } 
    
    #Values are loaded as string, so cast to float:
    history = {metric: list(map(float,history[metric])) for metric in history}

    #plot losses
    plt.figure(figsize=(14, 10), dpi=80)
    for metric in StyleGAN.METRICS:
        plt.plot(history[metric])
    plt.title("StyleGAN Training Losses")
    plt.xlabel("Epoch")

    #correctly display only integer epochs, with ticks starting from 1
    xlocs, xlabels = plt.xticks()
    xlocs = list(map(int, xlocs))
    xlabels = [x+1 for x in xlocs]
    plt.xticks(xlocs, xlabels)
    plt.xlim(xmin= 0, xmax=len(history[StyleGAN.METRICS[0]]))

    plt.ylabel("Loss")
    plt.legend(StyleGAN.METRICS)
    plt.savefig(output_file)
    
    #We employ this alternate method of showing image to prevent services 
    #potentially not being released properly, and performance issues with 
    #rendering such dense vectors.
    Image.open(output_file).show()
    
