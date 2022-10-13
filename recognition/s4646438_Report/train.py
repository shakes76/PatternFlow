from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from dataset import import_dataset, downsample_image
from modules import Model
upsample_factor = 4
batch_size = 8
read_image_size = (256, 240)
target_image_width = 256
epochs = 1

train, validation, test = import_dataset(batch_size, read_image_size, target_image_width, upsample_factor)

#load in the test image, pad and resize it to pass into the model
original_image = img_to_array(load_img(test[0]))
original_image = tf.image.pad_to_bounding_box(original_image, 0, 0, target_image_width, target_image_width)
test_image = np.expand_dims(original_image, axis=0)
original_image = array_to_img(original_image)
test_image = input_downsample(test_image, target_image_width, upsample_factor)
scaled = np.squeeze(test_image, axis=0)
scaled = array_to_img(tf.image.resize(scaled, (256, 256), method='bicubic'))

model = Model(upsample_factor)
#model.summary()
model.compile()
history = model.fit(train, epochs, validation, test_image=test_image)
