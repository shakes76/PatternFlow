class Model(object):

  def __init__(self, up_sample_factor):
    '''
    Initialises the model using the layers provided in the keras implementation
    in the task sheet (https://keras.io/examples/vision/super_resolution_sub_pixel/)
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

  def fit(self, training_dataset, epochs, validation_dataset):
    '''
    Fit the model to the dataset (train the model) with given number of epochs.
    Use validation_dataset to validate the model at each epoch.
    '''
    pass

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



  
