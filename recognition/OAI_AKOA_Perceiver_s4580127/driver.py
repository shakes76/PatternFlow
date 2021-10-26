import numpy as np
import tensorflow as tf
import glob
from PIL import Image

def load_data(batch_size):
    # Load images
    image_size = (128, 128)  # Original size - 260 * 228 TODO: should we keep the same ratio in HxW
    images = glob.glob("images/*.png")
    # TODO: this is currently storing all images in VRAM (2GB!!!) - should probs be changed to better batch handling
    images = np.array([np.array(Image.open(i).convert('L').resize(image_size)) for i in images])

    # Create 'channels' dimension for CNNs
    images = images[:, :, :, np.newaxis]
    discriminator_input_dim = images.shape[1:]
    dataset_size = images.shape[0]

    # Normalise images 0-1
    images = images/255

    # Create TensorFlow Dataset
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.shuffle(buffer_size=batch_size)
    images = images.repeat().batch(batch_size)
    image_iter = iter(images)

    return image_iter, discriminator_input_dim, dataset_size


if __name__ == '__main__':
    print(load_data(1))
