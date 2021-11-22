
import tensorflow as tf
import numpy as np
import PIL

# preprocess training images to array
def preprocess_array(imagelist):
    data = []
    for fname in imagelist:
        image = np.asarray(PIL.Image.open(fname))
        image = tf.image.resize(image, (256,256))
        data.append(image)
    data = np.array(data, dtype=np.float32)
    return data

# preprocess ground truth images to array
def preprocess_array_truth(imagelist):
    data = []
    for fname in imagelist:
        image = np.asarray(PIL.Image.open(fname))
        image = image[:,:,np.newaxis]
        image = tf.image.resize(image, (256,256), method = 'nearest')
        data.append(image)
    data = np.array(data, dtype=np.uint8)
    return data





