import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf_device='/gpu:0'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
a = tf.constant(10)
b = tf.constant(7)
c = tf.add(a, b)
print(c)
