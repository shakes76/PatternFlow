"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from predict import Visualiser



class Model(object):

  def __init__(self, up_sample_factor):
    '''
    Initialises the model using the layers provided in the keras implementation (REFERENCE 1)
    '''
    self._model = models.Sequential()
    self._model.add(layers.Conv2D(64, 5, input_shape=(None, None, 1), activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(64, 3, activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(32, 3, activation='relu', kernel_initializer='Orthogonal', padding='same'))
    self._model.add(layers.Conv2D(up_sample_factor **2, 3, activation='relu', kernel_initializer='Orthogonal', padding='same'))
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
    #TODO: implement function to load a single image path
    psnr = PSNR_Callback(None)
    model_callbacks = [early_stopping, checkpoints, psnr]
    return self._model.fit(training_dataset, validation_data=validation_dataset, initial_epoch=start_epoch, verbose=1, callbacks=model_callbacks)


  def load_weights(self):
    '''
    Load the (best) weights from training into the model.
    '''

    pass

  def predict(self, input_image):
    '''
    'Predict' the outout image (up-sample the input image)
    '''
    pass  


class PSNR_Callback(callbacks.Callback):
  '''
  Class which manages the computation of the mean peak signal to noise ratio of 
  images produced by a model in each epoch. This will be calculated after every epoch.
  '''
  def __init__(self, test_image=None, print_prediction_interval=15):
    '''
    initialise the PSNR callback.
    :param test_image: image used to visually inspect training progress every few epochs
    :param print_prediction_interval: (default 15) interval for current super 
    resolution prediction output with test_image. 1 means every epoch
    '''
    super(PSNR_Callback, self).__init__()
    self._test_image = test_image
    self._pred_interval = print_prediction_interval
    self._visialiser = Visualiser()
  
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
    print(f"Mean PSNR of epoch {epoch} is {np.mean(self._epoch_psnr)}.2f")
    if self._test_image != None and epoch % self._pred_interval:
      prediction = self._model.predict(self._test_image)
      self._visualiser.plot_results()


  def on_test_batch_end(self, batch, logs=None):
    '''
    Compute the peak signal to noise ratio for the current batch. (Calculation of PSNR taken
    from REFERENCE 1)
    '''
    self._epoch_psnr.append(10 * math.log10(1 / logs["loss"]))

  
  
