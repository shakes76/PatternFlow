```python
__author__ = 'xiaomeng cai'
__date__ = '10/15/2020 '

import tensorflow as tf
from tensorflow import keras

test_dir = r'E:\COMP3710DATA\dataset\validation'

test_datagen =keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# load model       
new_model = tf.keras.models.load_model('trained_model.h5')

loss, acc = new_model.evaluate(test_generator, verbose=2)
print('accuracy: {:5.2f}%'.format(100*acc))

```
