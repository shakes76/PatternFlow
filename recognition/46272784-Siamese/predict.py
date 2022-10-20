# This file contains functions for showing example usage of the trained model.
import os
import sys
sys.path.insert(1, os.getcwd())
import modules
import train
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
        print("Prediction: Alzheimer's disease")
    else:
        print("Prediction: Normal")
        
def main():
    siamese = modules.makeSiamese(modules.makeCNN())
    siamese.load_weights('siamese.h5')
    tr_a, tr_n, _, _, te_a, te_n = loadFile('F:/AI/COMP3710/data/AD_NC/')
    tr_a = tr_a.unbatch()
    tr_n = tr_n.unbatch()
    te_a = te_a.unbatch()
    te_n = te_n.unbatch()
    print(te_a.take(1).as_numpy_iterator())
    example = [tf.convert_to_tensor(list(tr_a.take(1).as_numpy_iterator())), tf.convert_to_tensor(list(tr_n.take(1).as_numpy_iterator()))]
    # print(example)
    if random.random() > 0.5:
        # Give a positive image 
        print('Known: AD')
        img = tf.convert_to_tensor(list(te_a.take(1).as_numpy_iterator()))
        predict(img, example, siamese)
    else:
        # Give a negative image 
        print('Known: NC')
        img = tf.convert_to_tensor(list(te_n.take(1).as_numpy_iterator()))
        predict(img, example, siamese)
    
if __name__ == '__main__':
    main()