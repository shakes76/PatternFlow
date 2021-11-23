import tensorflow as tf



'''

The two function retrived from:
https://github.com/keras-team/keras/issues/3611



'''
def dice_coefficient(y_true, y_pred):
    '''
    DSC=2TP/2TP+FP+FN
    
    
    '''
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())
def dice_loss(y_true, y_pred):
    '''
    Dice loss is based on the coefficient 

    '''
    

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator
