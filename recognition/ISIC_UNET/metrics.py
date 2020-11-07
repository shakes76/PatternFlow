"""
Module defining Dice Similarit Coefficient and related metrics.

@author: s4537175
"""

import tensorflow as tf

def dsc(true_segs, pred_segs):
	pred_flat = tf.keras.backend.flatten(pred_segs)
	true_flat = tf.keras.backend.flatten(true_segs)
	intersect = tf.keras.backend.sum(pred_flat * true_flat)
	return ( (2.0 * intersect) 
            / tf.keras.backend.sum(pred_flat + true_flat) )

def dsc_fore(true_segs, pred_segs):
    return dsc(true_segs[:,:,1], pred_segs[:,:,1])
    
def dsc_back(true_segs, pred_segs):
    return dsc(true_segs[:,:,0], pred_segs[:,:,0])

def avg_dsc(true_segs, pred_segs):
    return (0.5 * dsc_fore(true_segs, pred_segs) 
            + 0.5 * dsc_back(true_segs, pred_segs))

def dsc_loss(true_segs, pred_segs):
	return 1.0 - dsc(true_segs, pred_segs)

def avg_dsc_loss(true_segs, pred_segs):
    return (0.5 * dsc_loss(true_segs[:,:,0], pred_segs[:,:,0])
            + 0.5 * dsc_loss(true_segs[:,:,1], pred_segs[:,:,1]))


