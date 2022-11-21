"""
Reference: https://medium.com/@vedantjumle/image-generation-with-diffusion-
            models-using-keras-and-tensorflow-9f60aae72ac
"""

__author__ = "Zhao Wang, 46704847"
__email__ = "s4670484@student.uq.edu.au"


import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from modules import get_checkpoint, timesteps
from train import CKPT_PATH, IMAGE_SIZE
from tqdm import tqdm

# GIF saving path
GIF_PATH = "./gif/"

# create beta 
beta = np.linspace(0.0001, 0.02, timesteps)

# calculate alpha
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)

# load model
unet, ckpt_manager = get_checkpoint(CKPT_PATH)

# Save a GIF using logged images
def save_gif(img_list, path="", interval=200):
    """
    Saving pictures as .gif file.
    """
    # Transform images from [-1,1] to [0, 255]
    imgs = []
    for im in img_list:
        im = np.array(im)
        im = (im + 1) * 127.5
        im = np.clip(im, 0, 255).astype(np.int32)
        im = Image.fromarray(im)
        imgs.append(im)
    
    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs,
             save_all=True, duration=interval, loop=0)
    
def ddpm(x_t, pred_noise, t):
    """
    Taking the predicted noise off from the image in timestamp t.
    Parameters:
        x_t (np.array): the image in timestamp t.
        pred_noise (np.array): the predicted noise.
        t (int): the timestamp t.
    Returns:
        (np.array): the image in timestamp t-1.
    """
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    z = np.random.normal(size=x_t.shape)

    return mean + (var ** .5) * z

x = tf.random.normal((1,IMAGE_SIZE[0],IMAGE_SIZE[1],1))
img_list = []
img_list.append(np.squeeze(np.squeeze(x, 0),-1))

for i in tqdm(range(timesteps-1)):
    t = np.expand_dims(np.array(timesteps-i-1, np.int32), 0)
    pred_noise = unet(x, t)
    x = ddpm(x, pred_noise, t)
    img_list.append(np.squeeze(np.squeeze(x, 0),-1))

    if i % 200==0:
        plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255), 
                                                  np.uint8)[:,:,0], cmap="gray")
        plt.show()

save_gif(img_list + ([img_list[-1]] * 100), GIF_PATH, interval=20)
plt.imshow(np.array(np.clip((x[0] + 1) * 127.5, 0, 255), np.uint8)[:,:,0], 
                                                                    cmap="gray")
plt.show()