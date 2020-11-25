from pathlib import Path
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import json
import logging
import sys

matplotlib.use('Agg')

class Config:
    '''Configurable options for the DCGAN model'''
    image_size: int = 128
    image_channels: int = 1
    image_count: int = 11328
    seed_size: int = 100
    buffer_size: int = 100000
    batch_size: int = 128
    kernel_size: int = 5
    generator_alpha: float = 0.3
    generator_lr: float = 6e-05
    generator_beta1: float = 0.5
    discriminator_alpha: float = 0.3
    discriminator_lr: float = 4e-5
    discriminator_beta1: float = 0.5
    momentum: float = 0.8
    dropout: float = 0.25
    epochs: int =2
    checkpoint_freq: int = 10
    global_batch_size: int
    preview_margin: int = 15
    preview_cols: int = 6
    preview_rows: int = 4
    input_dir: Path = Path('input128')
    output_dir: Path = Path('output')

    fixed_seed: np.ndarray

    def __init__(self):
        self.fixed_seed = np.random.normal(0, 1, 
            (self.preview_rows * self.preview_cols, self.seed_size))

    def load_json(self, json_path: Path):
        '''
        Load a configuration from a json file and override defaults
        '''
		# TODO: Fix hacky
        with open(json_path, 'r') as f:
            data = json.load(f)

            for k, v in data.items():
                self.__dict__.__setitem__(k, v)
            if data.get('input_dir'):
                self.input_dir = Path(data['input_dir'])
            if data.get('output_dir'):
               self.output_dir = Path(data['output_dir'])

        return self
    
def set_logger(path: Path) -> logging.Logger:
    '''
    Set up logging
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger

def plot(loss_g: List[float], loss_d: List[float], d_real: List[float], 
        d_fake: List[float], plot_path: Path):
    '''
    Plot the loss functions for the discriminator and generator, and the
    output of the discriminator for the real and fake batches.
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))

    ax1.plot(loss_g, label="Generator Loss")
    ax1.plot(loss_d, label="Discriminator Loss")
    ax1.legend(loc='upper left')
    ax1.set_title('Generator and Discriminator Loss')

    ax2.plot(d_real, label="D(x)")
    ax2.plot(d_fake, label="D(G(z))")
    ax2.legend(loc='upper right')
    ax2.set_title('Discriminator Output')

    plt.savefig(plot_path)
    plt.close(f)
    return f, (ax1, ax2)


def preview_images(config: Config, images, epoch: int):
    '''
    Renders an image preview
    '''
    # Preview image
    margin = config.preview_margin
    rows = config.preview_rows
    cols = config.preview_cols
    size = config.image_size
    channels = config.image_channels

    image_array = np.full((
        margin + (rows * (size + margin)),
        margin + (cols * (size + margin))
        ), 255, dtype=np.uint8)

    images = 0.5 * images + 0.5

    i = 0
    for row in range(rows):
        for col in range(cols):
            r = row * (size + 16) + margin
            c = col * (size + 16) + margin
            image_array[r:r+size, c:c+size] = images[i].reshape(size, size) * 255
            i += 1

    im = Image.fromarray(image_array)

    return im

def hms_string(sec_elapsed):
    '''
    Converts seconds into a string with format hh:mm:ss.ms
    '''
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f'{h}:{m:02}:{s:05.2f}'

