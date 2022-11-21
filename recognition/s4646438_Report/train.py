from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import import_dataset
from modules import Model
from constants import epochs, checkpoint_loc, batch_size, read_image_size, target_image_width, upsample_factor


def plot_training_statistics(history, epochs):
  '''
  Produces plots to reflect loss, validation loss and psnr score during training
  :param history: keras history object (produced from model.fit)
  :param epochs: number of epochs that the model was trained for
  '''
  x = range(epochs)
  loss = history['loss']
  val_loss = history['val_loss']
  psnr = history['psnr']
  

  plt.plot(x, loss, label='loss')
  plt.plot(x, val_loss, label='val_loss')
  plt.legend()
  plt.savefig('training_statistics_loss.png')
  plt.show()
  plt.plot(x, psnr, label='psnr')
  plt.legend()
  plt.savefig('training_statistics_psnr.png')
  plt.ylim(0, 35)
  plt.savefig('training_statistics_psnr_range.png')
  plt.show()

#import the dataset
train, validation, test = import_dataset(batch_size, read_image_size, target_image_width, upsample_factor)

model = Model(upsample_factor)
model.summary()
model.compile()

history = model.fit(train, epochs, validation, checkpoint_loc=checkpoint_loc)
model_history = history.history

plot_training_statistics(model_history, epochs)
