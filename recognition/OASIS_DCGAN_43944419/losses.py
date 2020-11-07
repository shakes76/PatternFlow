'''
Losses module for Tensorflow

@author Peter Ngo

7/11/2020
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend

def generator_crossentropy(y_pred):

    """
    Calculate the binary cross entropy for the generator

    :param y_pred:
        The vector of outputs from the discriminator network on the generator's outputs.
    """
    return backend.mean(backend.binary_crossentropy(target=tf.ones_like(y_pred), output=y_pred, from_logits=True))

def discriminator_crossentropy(y1_pred, y2_pred):
    """
    Calculate the binary cross entropy for the discriminator

    :param y1_pred:
        The vector of outputs from the discriminator network on the generator's outputs.
    :param y2_pred:
        The vector of outputs from the discriminator network on the real brain images.
    """
    generated_ce = backend.mean(backend.binary_crossentropy(target=tf.zeros_like(y1_pred),output=y1_pred, from_logits=True))

    real_ce = backend.mean(backend.binary_crossentropy(target=tf.ones_like(y2_pred), output=y2_pred, from_logits=True))

    return generated_ce + real_ce

def discriminator_accuracy(y1_pred, y2_pred):
    """
    Calculate the binary cross entropy for the discriminator

    :param y1_pred:
        The vector of outputs from the discriminator network on the generator's outputs.
    :param y2_pred:
        The vector of outputs from the discriminator network on the real brain images.
    """
    #Create a tensor of all 0's with the same shape and type.
    generated_truth = tf.zeros_like(y1_pred)
    #Assign a value of 0.0 if the discriminator believes the fake image is fake (threshold of > 50%)
    generated_predictions = tf.cast(tf.map_fn(lambda x: 0.0 if x < 0.5 else 1.0, tf.keras.activations.sigmoid(y1_pred)), tf.float32)
    #Calculate discriminator accuracy on detecting fakes.
    gen_acc = backend.mean(backend.equal(generated_truth, generated_predictions))

    real_truth = tf.ones_like(y2_pred)
    #Assign a value of 1.0 if the discriminator believes the real image is real (threshold of > 50%)
    real_predictions = tf.cast(tf.map_fn(lambda x: 1.0 if x > 0.5 else 0.0, tf.keras.activations.sigmoid(y2_pred)), tf.float32)
    #Calculate discriminator accuracy on detecting real.
    real_acc = backend.mean(backend.equal(real_truth, real_predictions))
    
    return gen_acc, real_acc
