import tensorflow as tf
def dice_coefficient(truth, pred, smooth=1):
    """
    Code for this function written by Karan Jakhar (2019). Retrieved from:
    https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    """
    truth = tf.keras.backend.flatten(truth)
    pred = tf.keras.backend.flatten(pred)
    intersection = tf.keras.backend.sum(truth * pred)
    dice_coef = (2. * intersection + smooth) / (tf.keras.backend.sum(truth)
                                           + tf.keras.backend.sum(pred)
                                           + smooth)
    return dice_coef

def average_dice(data, model,test_size):
    #compute average dice
    input_batch, truth_batch = next(iter(data.batch(test_size)))
    predict = model.predict(input_batch)
    sum_DSC = 0
    for i in range(test_size):
        sum_DSC = sum_DSC + dice_coefficient(tf.argmax(truth_batch[i], axis=-1), tf.argmax(predict[i], axis=-1)).numpy()
    print(sum_DSC/test_size)