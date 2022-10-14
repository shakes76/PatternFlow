from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import import_dataset, downsample_image
from modules import Model
upsample_factor = 4
batch_size = 8
read_image_size = (256, 240)
target_image_width = 256
epochs = 20
do_training = True
checkpoint_loc = 'model/checkpoint'

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
  plt.show()
  plt.plot(x, psnr, label='psnr')
  plt.ylim(0, 35)
  plt.legend()
  plt.show()


train, validation, test = import_dataset(batch_size, read_image_size, target_image_width, upsample_factor)

model = Model(upsample_factor)
model.summary()
model.compile()

history = model.fit(train, epochs, validation, checkpoint_loc=checkpoint_loc)
model_history = history.history

plot_training_statistics(model_history, epochs)
