# This file contains functions for showing example usage of the trained model.
import os
import sys
sys.path.insert(1, os.getcwd())
import modules
from dataset import loadFile
import random
import tensorflow as tf
from tensorflow import keras

def predict(test_image, examples, siamese):
    """Give prediction on whether the brain in the image has alzheimers

    Args:
        test_image: the target image
        examples: A list [ad, nc], where ad is a known ad image, nc is a known nc image (from training dataset)
        siamese: the siamese network
    """
    if len(examples) != 2:
        print('Usage: give only two examples [ad_img, nc_img]')
    ad_score = siamese([test_image, examples[0]], training=False)
    nc_score = siamese([test_image, examples[1]], training=False)
    if ad_score > nc_score:
        return 1
    else:
        return 0
        
def random_predict_one_shot(n=600):
    """
    Perform 1-shot image classification
    """
    count = 0
    siamese = modules.makeSiamese(modules.makeCNN())
    siamese.load_weights('siamese.h5')
    tr_a, tr_n, _, _, te_a, te_n = loadFile('F:/AI/COMP3710/data/AD_NC/')
    tr_a = tr_a.unbatch()
    tr_n = tr_n.unbatch()
    te_a = te_a.unbatch()
    te_n = te_n.unbatch()
    for i in range(n):
        if i % 20 == 0:
            print(">> Test {}".format(i))
        # randomly select known images
        example = [tf.convert_to_tensor(list(tr_a.take(1).as_numpy_iterator())), tf.convert_to_tensor(list(tr_n.take(1).as_numpy_iterator()))]
        if random.random() > 0.5:
            # Give a positive image 
            img = tf.convert_to_tensor(list(te_a.take(1).as_numpy_iterator()))
            if predict(img, example, siamese) == 1:
                count += 1
        else:
            # Give a negative image 
            img = tf.convert_to_tensor(list(te_n.take(1).as_numpy_iterator()))
            if predict(img, example, siamese) == 0:
                count += 1
    return count / n
        

def main():
    print(">> Test accuracy: {}".format(random_predict_one_shot()))
    
if __name__ == '__main__':
    main()