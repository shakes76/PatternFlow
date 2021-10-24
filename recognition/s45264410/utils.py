import os
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('buffer_size', 400, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_string('path', None, 'Path to the data folder')
flags.DEFINE_boolean('enable_function', True, 'Enable Function?')

IMG_WIDTH = 128
IMG_HEIGHT = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_images(image_file):
    """Loads the image and generates input and target image.
    Args:
        image_file: .jpeg file
    Returns:
        Input image, target image
    """
    
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    input_image, input_mask = normalize(input_image, )

    return input_image, input_mask



