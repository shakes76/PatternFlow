# This file contains the source code for training, validating, testing and saving my model
import os
import sys
sys.path.insert(1, os.getcwd())
import modules
from dataset import loadFile
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
import numpy as np
import random

def getOptimizer():
    return keras.optimizers.Adam(1e-4)

def saveOption(optimizer, siamese):
    checkpoint_dir = os.path.join(os.getcwd(), "Siamese_ckeckpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                  net=siamese)
    return checkpoint_prefix, checkpoint

@tf.function
def train_step(pairs, optimizer, siamese):
    with tf.GradientTape() as gra_tape:
        lossValue = (modules.loss())(pairs[2], siamese([pairs[0], pairs[1]]))
        # lossValue = tf.keras.losses.BinaryCrossentropy(from_logits=True)(pairs[2], siamese([pairs[0], pairs[1]]))
        gradient = gra_tape.gradient(lossValue, siamese.trainable_weights)
        optimizer.apply_gradients(zip(gradient, siamese.trainable_weights))
    return lossValue

def train(dataset, epochs, train_step, checkpoint_prefix, checkpoint, optimizer, siamese):
    lossInfo = []
    for epoch in range(epochs):
        start = time.time()
        print('>>>>>>>>> Epoch {}'.format(epoch+1))
        count = 0
        siameseLoss = 0
        with tqdm(dataset, unit='batch') as tepoch:
            for pair_batch in tepoch:
                lossValue = train_step(pair_batch, optimizer, siamese)
                siameseLoss += lossValue
                count += 1
                tepoch.set_postfix(loss=lossValue.numpy())
            
        # Save the model every epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        lossInfo.append(siameseLoss / count)
        

    return lossInfo

def main():
    t, v = loadFile('F:/AI/COMP3710/data/AD_NC/')
    td = modules.generatePairs(t)
    vd = modules.generatePairs(v)
    opt = getOptimizer()
    siamese = modules.makeSiamese(modules.makeCNN())
    checkpoint_prefix, checkpoint = saveOption(opt, siamese)
    history = train(td, 5, train_step, checkpoint_prefix, checkpoint, opt, siamese)
    
    # # results = siamese.evaluate(vd)
    # # print("test loss, test acc:", results)
    
if __name__ == "__main__":
    main()