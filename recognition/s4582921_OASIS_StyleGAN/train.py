import matplotlib.pyplot as plt
import tensorflow as tf
import glob

IMAGE_SIZE = 64
BATCH_SIZE = 32
IMAGE_PATHS = glob.glob('./keras_png_slices_data/keras_png_slices_test/*.png') \
    +  glob.glob('./keras_png_slices_data/keras_png_slices_validate/*.png') \
    +  glob.glob('./keras_png_slices_data/keras_png_slices_train/*.png')

image_tensor = tf.convert_to_tensor(IMAGE_PATHS, dtype=tf.string)
dataset = tf.data.Dataset.from_tensor_slices(image_tensor)

def preprocessing(path):
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

dataset = dataset.map(preprocessing, num_parallel_calls=8).cache('./cache/')
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(BATCH_SIZE)

for batch in dataset:
    plt.imshow(batch[0][:,:,0])
    plt.show()
    break
