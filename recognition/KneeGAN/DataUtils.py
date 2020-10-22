import tensorflow as tf
from PIL import Image
import glob
import numpy as np

def load_data(filepath, batch_size):
    image_files = glob.glob(filepath + '*')
    #images = np.array([np.array(Image.open(i)) for i in image_files])
    crop = (30, 55, 150, 175)
    images = np.array([np.array((Image.open(i).crop(crop)).resize((64,64))) for i in image_files[:500]])

    discriminator_input_dim = images.shape[1:]
    dataset_size = images.shape[0]

    images = images/255

    print("Data Shape:")
    print(images.shape)

    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.shuffle(buffer_size=batch_size)
    images = images.repeat().batch(batch_size)
    image_iter = iter(images)

    return image_iter, discriminator_input_dim, dataset_size
