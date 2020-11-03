'''
Losses module for Tensorflow
'''
import tensorflow as tf
from tensorflow import keras
from keras import backend

def generator_crossentropy(y_pred):
    
    #return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
	return backend.binary_crossentropy(target=tf.ones_like(y_pred), output=y_pred, from_logits=True)

def discriminator_crossentropy(y1_pred, y2_pred):

    generated_ce = backend.binary_crossentropy(target=tf.zeros_like(y1_pred), 
                                                    output=y1_pred, 
                                                    from_logits=True)

    real_ce = backend.binary_crossentropy(target=tf.ones_like(y2_pred), 
                                                output=y2_pred, 
                                                from_logits=True)

    return generated_ce + real_ce