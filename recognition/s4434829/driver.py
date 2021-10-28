"""
Driver script to show an example usage of the model

TODO: unsure if training can be here or needs to be separate
or maybe training needs to be in the class?
can I have other files?
From ed: yeah training can be in here thats ok
        maybe ake it a funciton at least
        also probabyl move dataloader, especially if will have more than one

TODO:
they reccomend starting with minst
- add rest of vq
- train with minst to see how good reconstructions are and make sure works
- then either train on data, later add prior for generation
    or add prior and test with misnt then train with data
    not sure if can use same trained after ahve prior
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pathlib
import math
import time

from model import VQVAE
from data_loaders import load_oasis_data, load_minst_data


def main():
    print("hello")
    ### load data
    # train_seq, valid_seq, test_seq = load_oasis_data(path="./keras_png_slices_data/keras_png_slices_data")
    # X_train, y_train = train_seq.__getitem__(1)
    # # print(X_train.__len__())
    # print(tf.image.ssim(X_train[0], X_train[0], 3))
    # print(tf.image.ssim(X_train[0], X_train[1], 3))

    # plt.imshow(X_train[0], cmap='gray')
    # plt.figure()
    # plt.imshow(y_train[0], cmap='gray')
    # plt.show()
    # width, height = 256, 256

    # load minst data
    width, height = 28, 28
    train_seq, test_seq = load_minst_data(batch_size=128)


    ### initilise model
    model = VQVAE()
    model.build( input_shape=[10, width, height, 1])
    print(model.summary())

    ### make optimiser and loss
    lr = 2*10**(-4)
    batch_size = 128
    optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

    ### train model
    # Some of this I reuse from my demo code but i think that's ok it wouldnt 
    # change much anyway. It does use the loss returned instead of recalcualting loss
    # TODO move
    loss_hist = [] # for plotting
    EPOCHS = 5
    idx = 0
    for epoch in range(EPOCHS):
        start = time.time()
        for sample, target in train_seq:
            idx+=1
            with tf.GradientTape() as tape:
                internal_loss, out = model(sample, training=True) 
    #             print(sample.shape, out.shape, internal_loss.shape)
                # need to take some measure of closeness of image into accoutn
                # TODO: try another loss
                image_dif = 1- tf.image.ssim(sample, out, 3)
                loss = internal_loss + image_dif 
    #             loss, out = model(sample, training=True) 
                
    #             loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target,out)

            # step of optamisiser
            grads = tape.gradient(loss, model.trainable_weights)
            optimiser.apply_gradients((zip(grads, model.trainable_weights)))
            
            loss_hist += [np.mean(loss)]
            if (idx%100 == 0):    
                print('Run {}: \n\ttrain loss: {}'.format(idx, np.mean(loss)))

            if (idx%300 == 0):
                folder = "./saves/initial/"
                time_now = time.time()
                model_name = "model_1_{}".format(time_now)
                model.save_weights(folder+model_name+"model_weights")
                np.save("./saves/initial/losses/loss_{}".format(time_now), loss_hist, allow_pickle=True) 
            
        print("time for one epoch: ",time.time()-start) 

        if(epoch % 1 == 0):
            print('Epoch {}: \n\ttrain loss: {}'.format(epoch, np.mean(loss)))


    ### plot results

    print("Done")


if __name__ == "__main__":
    main()
