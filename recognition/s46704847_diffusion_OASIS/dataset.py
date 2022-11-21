# -*- coding: utf-8 -*-

__author__ = "Zhao Wang, 46704847"
__email__ = "s4670484@student.uq.edu.au"

import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf

def get_zipped_dataset(path, image_size):
  """
  Reading zipped dataset from the path.
  Parameters:
    path (str): the path of the dataset.
  Returns:
    train_images (np.array): the images in the np.array type.
  """
  zipped_images = zipfile.ZipFile(path)
  image_list = zipped_images.namelist()
  image_list = [image for image in image_list if '.png' in image]
  train_images = np.array([np.array(Image.open(zipped_images.open(image))
                                  .resize(image_size)) for image in image_list])
  return train_images

def normalize(dataset):
  """
  Normalizing the np.array image into [-1,1] and then adding a single channel.
  Parameters:
    dataset (np.array): the initial images.
  Return:
    trian_images (np.array): the normalized images.
  """
  train_images = dataset / 127.5 - 1
  train_images = train_images[:,:,:,np.newaxis]
  return train_images

def set_train_batch(dataset, batch_size):
  """
  Splitting the whole dataset into batch dataset.
  Parameters:
    dataset (np.array): the initial dataset.
    batch_size (int): the size of the batch.
  Returns:
    (tf.Tensor): the batch of dataset.
  """
  length = len(dataset)
  return  tf.data.Dataset.from_tensor_slices(dataset).shuffle(length)\
                                                      .batch(batch_size)
