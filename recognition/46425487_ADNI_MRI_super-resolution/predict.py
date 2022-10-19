import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.models.load_model('./trained_model')
test_ds = tf.data.Dataset.load('./test_data')

for batch in test_ds.take(1):
    for i, low_res in enumerate(batch[0]):
        high_res = batch[1][i]
        inp = np.expand_dims(low_res, axis=0)
        print(inp.shape)
        predicted = model.predict(inp)
        predicted = predicted[0]
        print(low_res.shape)
        print(high_res.shape)
        print(predicted.shape)