import tensorflow.keras.backend as K
def dice_coefficient(a, b):
    """
    Dice Coefficient function from : https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Used to determine how closely two set overlap each other. In this case we use it to see how close the
    predicted mask matches the ground truth mask.
    """
   
    
    a = K.flatten(a)
    b = K.flatten(b)
    a_union_b = K.sum(a * b)
    mag_a = K.sum(a)
    mag_b = K.sum(b)
    
    return (2.0 * a_union_b) / (mag_a + mag_b)

def dice_coefficient_loss(truth, predition):
    """
    Loss function as described in the Improved Unet paper.
    """
    return 1 - dice_coefficient(truth, predition)