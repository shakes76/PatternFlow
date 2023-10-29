def dice_coefficient(y_true, y_pred, smooth=0.0001):
    #change the dimension to tree
    y_true_f =tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    #calculation for the loss function
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1. - dice_coefficient(y_true, y_pred)