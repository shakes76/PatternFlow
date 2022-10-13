from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from dataset import import_dataset, input_downsample
from modules import Model
upsample_factor = 4
batch_size = 8
read_image_size = (256, 240)
target_image_width = 256
epochs = 50

train, validation, test = import_dataset(batch_size, read_image_size, target_image_width, upsample_factor)

#load in the test image, pad and resize it to pass into the model
test_image_init = img_to_array(load_img(test[0]))
test_image_init = tf.image.pad_to_bounding_box(test_image_init, 0, 0, target_image_width, target_image_width)
test_image = np.expand_dims(test_image_init, axis=0)
test_image = input_downsample(test_image, target_image_width, upsample_factor)
test_image = np.squeeze(test_image, axis=0)

model = Model(upsample_factor)
model.summary()
model.compile()
history = model.fit(train, epochs, validation, test_image=test_image)
