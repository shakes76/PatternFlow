'''
Losses module for Tensorflow
'''
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.activations import sigmoid

def generator_crossentropy(y_pred):
    
    
	return backend.mean(backend.binary_crossentropy(target=tf.ones_like(y_pred), 
                                                    output=y_pred, 
                                                    from_logits=True))

def discriminator_crossentropy(y1_pred, y2_pred):

    generated_ce = backend.mean(backend.binary_crossentropy(target=tf.zeros_like(y1_pred), 
                                                    output=y1_pred, 
                                                    from_logits=True))

    real_ce = backend.mean(backend.binary_crossentropy(target=tf.ones_like(y2_pred), 
                                                output=y2_pred, 
                                                from_logits=True))
    return generated_ce + real_ce

def discriminator_accuracy(y1_pred, y2_pred):
    generated_truth = tf.zeros_like(y1_pred)
    generated_predictions = tf.cast(tf.map_fn(lambda x: 0.0 if x < 0.5 else 1.0, sigmoid(y1_pred)), tf.float32)
    gen_acc = backend.mean(backend.equal(generated_truth, generated_predictions))

    real_truth = tf.ones_like(y2_pred)
    real_predictions = tf.cast(tf.map_fn(lambda x: 1.0 if x > 0.5 else 0.0, sigmoid(y2_pred)), tf.float32)
    real_acc = backend.mean(backend.equal(real_truth, real_predictions))
    
    return gen_acc, real_acc