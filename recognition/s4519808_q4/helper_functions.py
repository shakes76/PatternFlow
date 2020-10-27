"""
COMP3710 Report 

This file contains helper functions such as dice similarity, image resize, ....

@author Huizhen 
"""

import tensorflow as tf
import numpy as np

def resize_image(image, h, w):
    """ resize the image to (h, w, channel)"""
    new_image = tf.image.resize(image, [h, w]).numpy()  
    return new_image


# Define Dice Similarity Coefficient
def dsc(y_true, y_pred):
    """
    y_true, y_pred : groundtruth and prediction images in numpy array format.
    """
    intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
    dsc = 2*intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dsc

def dsc_loss(y_true, y_pred):
    return 1.0 - dsc(y_true, y_pred)







############################# Not used functions related to batch data genertor #########################
########################### some of them might be wrong sperated from the main.py #######################

def load_batch(input_images: list, output_images: list, h, w, batch_size=1):
    """
    input_images : image name list (str)
    output_images : image name list (str)
    """
    idx = 0
    i = []
    o = []
    while 1:
        while 1:
            i.append(resize_image(mpimg.imread(input_images[idx])/255,h,w))
            o.append(resize_image(mpimg.imread(output_images[idx])[:,:,np.newaxis],h,w))
            idx += 1
            if len(i) == batch_size: 
                yield (np.array(i), np.array(o))
                i = []
                o = []
            if idx == len(input_images):
                idx = 0

def avg_dsc():
    """
    Calculate average DSC value
    """
    test = load_batch(test_input_images, test_output_images, H, W)
    avg_dsc = 0
    for _ in range(len(test_input_images)):
        img = next(test)
        pred = model.predict(img[0][0][np.newaxis,:,:,:])
        predd = tf.math.round(pred)
        avg_dsc += dsc(predd[0], img[1][0])

    return avg_dsc/len(test_input_images)

def compare_segment(fail_case, dsc_threshold):
    """
    Help find predictions with a specific DSC value.
    """
    while 1:
        plot = False
        img = next(test)
        pred = model.predict(img[0][0][np.newaxis,:,:,:])
        predd = tf.math.round(pred)
        dsc_score = dsc(predd[0], img[1][0])
        
        if fail_case:
            if dsc_score <= dsc_threshold:
                plot = True
                
        if not fail_case:
            if dsc_score >= dsc_threshold:
                plot = True
        
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14,4))
            fig.suptitle(f"Input,                GroundTruth,               Prediction DSC: {dsc_score}")
            ax1.imshow(img[0][0])
            ax2.imshow(img[1][0])
            ax3.imshow(tf.math.round(model.predict(img[0][0][np.newaxis,:,:,:]))[0])
            break
