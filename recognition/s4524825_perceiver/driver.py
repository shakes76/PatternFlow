import numpy as np
# from recognition.s4524825_perceiver.train import correct_num_batch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import math
import data as d
import config as c
from perceiver import Perceiver
import json

def correct(y_true, y_pred):
    pred = tf.argmax(y_pred, -1)
    pred = tf.reshape(pred, (min(c.batch_size, pred.shape[0]), 1))
    pred = tf.cast(pred, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)

    correct_num = tf.equal(y_true, pred)
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return (correct_num > 0).numpy()

testing_it = d.test_data_iterator()

model = Perceiver()

#call to initialize
model(np.zeros((1, 64, 64, 3)))

model.load_weights("goliath/weights_old.h5")

images, labels = testing_it.next()

image = images[0]
label = labels[0]

im = image.numpy()
for i in range(3):
    im[..., i] = ((im[..., i] * 127.5) + 127.5)

plt.imshow(im.astype(np.int))

label_to_content = {}
with open("label_to_content.json", 'r') as f:
    label_to_content = json.load(f)


print("#######################################")
prediction = model([image], training=False) #pad for batch size of 1
print(prediction)
label_num = np.argmax(prediction)
print(f"predict (label num = {label_num}) which corresponds to {label_to_content[str(label_num)]}")
print(f"predicted correct label? {correct([label], prediction)}")
print(f"true prediction = {label_to_content[str(int(label[0].numpy()))]}")

print("######################################")
plt.show()