"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""

import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import array_to_img

class Model(object):

  def __init__(self, up_sample_factor):
    '''
    Initialises the model using the layers provided in the keras implementation (REFERENCE 1)
    '''
    self._model = models.Sequential()
    self._model.add(layers.Conv2D(64, 4, input_shape=(None, None, 1), activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(64, 2, activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(32, 2, activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(up_sample_factor **2, 2, activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Lambda(lambda x: tf.nn.depth_to_space(x, up_sample_factor)))
    

  def summary(self):
    '''
    Produces a summary of the models layers. This is used to check that it has
    been created properly.
    '''
    self._model.summary()

  def compile(self):
    '''
    Compiles the model with Adam optimiser (lr=0.001) and MSE loss.
    '''
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = loss_fn = keras.losses.MeanSquaredError()
    self._model.compile(optimizer=optimizer, loss=loss)

  def fit(self, training_dataset, epochs, validation_dataset, checkpoint_loc='model/checkpoint', start_epoch=0):
    '''
    Fit the model to the dataset (train the model) with given number of epochs.
    Use validation_dataset to validate the model at each epoch.
    The following callbacks are defined
      - early stopping (the model will stop after 8 epochs with constant loss)
      - checkpoints (the weights of the best epoch will be saved to checkpoint_loc)
      - PSNR (computes the peak signal to noise ratio, the metric commonly used to evaluate superresolution networks)

    :param training_dataset: the dataset to train on (this is a tensorflow dataset)
    :param epochs: number of epochs to train with
    :param start_epoch: (default 0) epoch to begin training on
    :param checkpoint_loc: (default 'checkpoint') file path for best checkpoint weights to be saved at
    :return: history object which can then be used to plot training statistics
    '''
    #if the loss remains the same for patience=8 epochs, the training will be 
    #terminated early and the best weights will be loaded into the model (for predictions)
    early_stopping = callbacks.EarlyStopping(patience=8, monitor='loss', restore_best_weights=True)

    #saves the weights of the best checkpoint so far
    checkpoints = callbacks.ModelCheckpoint(checkpoint_loc,
                                            monitor='loss',
                                            save_best_only=True,
                                            save_weights_only=True)
    
    psnr = PSNR_Callback()
    model_callbacks = [early_stopping, checkpoints, psnr]
    return self._model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset, initial_epoch=start_epoch, verbose=1, callbacks=model_callbacks)


  def load_weights(self, checkpoint_loc):
    '''
    Load the (best) weights from training into the model.
    '''
    self._model.load_weights(checkpoint_loc)



  def predict(self, input_image):
    '''
    'Predict' the output image (up-sample the input image)
    :param input_image: scaled down version of the image array of shape(1, x, x, 1)
    :return: upscaled image according to the trained model - PIL image object (x, x, 1)
    '''
    pred = self._model.predict(input_image)
    #remove the extra axis to turn into a PIL image
    pred = np.squeeze(pred, axis=0)
    return array_to_img(pred)




class PSNR_Callback(callbacks.Callback):
  '''
  Class which manages the computation of the mean peak signal to noise ratio of 
  images produced by a model in each epoch. This will be calculated after every epoch.
  '''
  def __init__(self):
    '''
    initialise the PSNR callback.
    '''
    super(PSNR_Callback, self).__init__()
  
  def on_epoch_begin(self, epoch, logs=None):
    '''
    Store the peak signal to noise ratio for each batch
    '''
    self._epoch_psnr = []

  def on_epoch_end(self, epoch, logs=None):
    '''
    Output the mean peak signal to noise ratio at the end of each epoch. Allows training
    'metrics' to be monitored during training. Output image with zoomed component every
    print_prediction_interval epochs.
    '''
    mean = np.mean(self._epoch_psnr)
    print(f"\nMean PSNR of epoch {epoch} is {mean:.10f}")
    logs['psnr'] = mean

  def on_test_batch_end(self, batch, logs=None):
    '''
    Compute the peak signal to noise ratio for the current batch. (Calculation of PSNR taken
    from REFERENCE 1)
    '''
    self._epoch_psnr.append(10 * math.log10(1 / logs["loss"]))

  
  
