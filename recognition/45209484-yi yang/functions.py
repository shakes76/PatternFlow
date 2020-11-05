import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K 
import PIL

# convert training images to array
def convert_array(filelist):
    data = []
    for fname in filelist:
        image = np.asarray(PIL.Image.open(fname))
        image = tf.image.resize(image, (256,256))
        data.append(image)
    data = np.array(data, dtype=np.float32)
    return data

# convert ground truth images to array
def convert_array_truth(filelist):
    data = []
    for fname in filelist:
        image = np.asarray(PIL.Image.open(fname))
        image = image[:,:,np.newaxis]
        image = tf.image.resize(image, (256,256), method = 'nearest')
        data.append(image)
    data = np.array(data, dtype=np.uint8)
    return data

# compile and fit model
def fit(model,x,y, epoch_size, batch):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                     metrics=['accuracy'])

    model.fit(x, y, epochs=epoch_size, batch_size=batch,
                    validation_split=0.2)
