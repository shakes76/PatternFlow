from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import tensorflow as tf

# -----------------------
# load data from given path and pre processing
# -----------------------
def load_dataset(root_dir, batch_size):
  """
  load data from given path, resize images and assign in batch, devide training 
  set and validation set.
  :param root_dir: the path of dataset
  :param batch_size: the number of images in a batch
  :param crop_size: resize original image to crop size
  :return: train_ds(0.9) and valid_ds(0.1)
  """
  train_ds = image_dataset_from_directory(
      root_dir,
      batch_size=batch_size,
      image_size=(crop_size, crop_size),
      validation_split=0.1,
      subset="training",
      seed=1337,
      label_mode=None,
      )

  valid_ds = image_dataset_from_directory(
      root_dir,
      batch_size=batch_size,
      image_size=(crop_size, crop_size),
      validation_split=0.1,
      subset="validation",
      seed=1337,
      label_mode=None,
      )
  return train_ds, valid_ds

# -----------------------
# normalization
# -----------------------
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

# -----------------------
# load test set
# -----------------------
def test_imgs(test_dir):

  test_path = os.path.join(test_dir)

  test_img_paths = sorted(
      [
          os.path.join(test_path, fname)
          for fname in os.listdir(test_path)
          if fname.endswith(".jpeg")
      ]
  )
  return test_img_paths


# -----------------------
# low resolution images
# -----------------------
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

# -----------------------
# high resolution images
# -----------------------
def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y
    
# -----------------------
# prepare training images
# -----------------------
def dataset_preprocessing(train_ds,valid_ds):

    train_ds = train_ds.map(scaling)
    valid_ds = valid_ds.map(scaling)
    train_ds = train_ds.map(
        lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
    )
    train_ds = train_ds.prefetch(buffer_size=64)
    valid_ds = valid_ds.map(
        lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
    )
    valid_ds = valid_ds.prefetch(buffer_size=64)
    return train_ds, valid_ds