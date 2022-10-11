"""
Source code for training, validating, testing, and saving alzheimers classification model.
"""
import tensorflow as tf

tf.get_logger().setLevel('INFO')
assert len(tf.config.list_physical_devices("GPU")) >= 1, "No GPUs found"
