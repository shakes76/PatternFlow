from PIL import Image
import numpy as np
import tensorflow as tf
import os
import shutil
import csv


def denormalise(data: np.array, mean: float) -> np.array:
    """
    Take output of GAN and undo data preprocessing procedure to return
    a matrix of integer greyscale image pixel intensities suitable to
    convert directly into an image.

    Args:
        data (np.array): normalised data
        mean (float): grop mean used to centralise original normalised data

    Returns:
        np.array: denormalized data
    """
    data = np.array(data) #cast to numpy array from any array like (Allows Tensor Compatibility)
    decentered = data + mean
    return (decentered * 255).astype(np.uint8)


def create_image(data: np.array, filename: str = None) -> Image:
    """
    Creates an new PNG image from a generated data matrix.
    Saves image to output folder if a name is specified

    Args:
        data (np.array): uint8 single channel matrix of greyscale image pixel intensities
        name (str or NoneType, optional): filename and path to save image, If None image is not saved. Defaults to None.
        output_folder (str, optional): path of output folder. Defaults to "output/".

    Returns:
        Image: Generated image
    """
    im = Image.fromarray(data[:,:,0],'L').convert("RGBA") #Passed data will have extreneous channel dimension
    if filename is not None:
        im.save(filename+".png")
    return im

def random_generator_inputs(num_images: int,latent_dim: int,noise_start: int,noise_end: int) -> list[np.array]:
    """_summary_ TODO

    Args:
        num_images (int): _description_
        latent_dim (int): _description_
        noise_start (int): _description_
        noise_end (int): _description_

    Returns:
        list[np.array]: _description_
    """

    latent_vectors = tf.random.normal(shape = (num_images,latent_dim))
    input_tensors = [latent_vectors]
    curr_res = noise_start
    while curr_res <= noise_end:
        input_tensors.append(tf.random.normal(shape = (num_images,curr_res,curr_res,latent_dim)))
        input_tensors.append(tf.random.normal(shape = (num_images,curr_res,curr_res,latent_dim)))
        curr_res = curr_res*2
    return input_tensors

def make_fresh_folder(folder_path: str) -> None:
    """
    Generates a folder at the specified location, wiping any existing contents
    If the specified folder already existed. Parents are created if neccessary,
    but are not cleared if they exist already.

    Args:
        folder_path (str): path to specified folder. If no folder exists at the location a new one is created. If a folder exists at the location it is wiped.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def save_training_history(history: dict[list[float]], filename: str) -> None: #TODO docsttinfs
     with open(filename + ".csv", mode = 'a') as f:
        csv.writer(f).writerows(zip(*history.values())) #we pass in the arbitrary length set of *args (various history compoents)

def load_training_history(csv_location: str) -> dict[list[float]]:
    pass

def plot_training(history: dict[list[float]], output_folder: str, epoch_range: tuple[int,int] = None) -> None:
    pass