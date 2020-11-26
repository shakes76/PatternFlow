"""
Module defining Dice Similarity Coefficient metrics.
"""

import tensorflow as tf

def dsc(true_segs, pred_segs):
    """
    Returns the dice similariy coefficient (DSC) between two sets.
    DSC is two times the intersection of the sets, dived by the sum of
    the sizes of the two sets.
    """
    # Flatten both tensors into vectors
    pred_flat = tf.keras.backend.flatten(pred_segs)
    true_flat = tf.keras.backend.flatten(true_segs)
    # Intersection is sum of element-wise multiplication as segmentation
    # results are probabilities between 0 and 1
    intersect = tf.keras.backend.sum(pred_flat * true_flat)
    # Denominator is sum elements in both vectors
    return (2.0 * intersect)/tf.keras.backend.sum(pred_flat + true_flat)

def dsc_fore(true_segs, pred_segs):
    """
    Returns DSC of the foreground segmentation result.
    """
    return dsc(true_segs[:,:,1], pred_segs[:,:,1])
    
def dsc_back(true_segs, pred_segs):
    """
    Returns DSC of the background segmentation result.
    """
    return dsc(true_segs[:,:,0], pred_segs[:,:,0])

def avg_dsc(true_segs, pred_segs):
    """
    Returns average DSC of the foreground and background segmentation results.
    """
    return (0.5 * dsc_fore(true_segs, pred_segs) 
            + 0.5 * dsc_back(true_segs, pred_segs))

def dsc_loss(true_segs, pred_segs):
    """
    Returns the DSC loss. i.e. 1 - DSC
    """
    return 1.0 - dsc(true_segs, pred_segs)

def avg_dsc_loss(true_segs, pred_segs):
    """
    Returns average DSC loss of the foreground and background segmentation
    results.
    """
    return (0.5 * dsc_loss(true_segs[:,:,0], pred_segs[:,:,0])
            + 0.5 * dsc_loss(true_segs[:,:,1], pred_segs[:,:,1]))
