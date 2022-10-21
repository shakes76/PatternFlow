"""
Driver script to show an example usage of the model
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pathlib
import math
import time

from model import VQVAE, PixelCNN
from data_loaders import * 

def gen_images(pixel_cnn, vq, decoder, indices_shape, num):
    """
    Generate images using the pixelcnn
    
    Source: the code for this comes from here: 
    https://keras.io/examples/generative/vq_vae/#codebook-sampling
    although I use embedding_lookup which makes it a bit simpler. Most
    notably making a model specifically to sample the distribution comes from here 
    because the rest of it was more obvious but sampling a distribution in tf was 
    new to me.
    Also this explanation https://bjlkeng.github.io/posts/pixelcnn/ helped me 
    understand how to go from pixelcnn to images.
    """
    # Make a sampler model that takes the output of pixel_cnn
    # and use it as the probabilities in a catagorical ditribution
    # So when we sample the  distribution it's catagories and probabilities for
    # outputs come from the output of pixel_cnn
    # In this way we are making random samples with the same distribution
    # as the pixel cnn
    # this comes from https://keras.io/examples/generative/vq_vae/#codebook-sampling
    inputs = tf.keras.layers.Input(shape=indices_shape, dtype=tf.int32)
    idxs = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=idxs)
    sampled = dist.sample()
    sampler = tf.keras.Model(inputs, sampled)

    # to generate an image we need to sample the distribution for a set of pixels
    # however the probabilty is conditional, so every pixel depends on all
    # other pixels before and to the left of it
    # therefore we need to create an empty set of pixels, then go through
    # each pixel from the top left, generating each subsequenct pixel based on 
    # all previous pixels.

    # create a set of empty pixels with the same shape as the output of the sampleer
    priors = np.zeros(shape=(num,) + (pixel_cnn.input_shape)[1:])
    num, rows, cols = priors.shape

    # iterate through rows and columns, using the previously sampled priors
    # as the input for sampler model
    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors)
            # update values only for this row and column
            priors[:, row, col] = probs[:, row, col]

    # Now we need to do the reverse part of the vqvae from after calculting
    # indices. We use the generated rpriors as the indices and then decode
    quantised = tf.nn.embedding_lookup(tf.transpose(vq.emb, [1, 0]), priors.astype("int32"))
    # print(quantised.shape)
    generated = decoder.predict(quantised)
    return generated


def main():
    print("hello")
    ### load data
    width, height = 256, 256
    train_seq, valid_seq, test_seq = load_oasis_data("./keras_png_slices_data/keras_png_slices_data", batch_size=128)

    # load minst data
    # width, height = 28, 28
    # train_seq, test_seq = load_minst_data(batch_size=128)

    ### initilise model
    model = VQVAE()
    model.build(input_shape=[1, width, height, 1])
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
    EPOCHS = 0 # NOTE SKIPPING but better to make it a function
    idx = 0
    for epoch in range(EPOCHS):
        start = time.time()
        for sample, target in train_seq:
            idx+=1
            with tf.GradientTape() as tape:
                internal_loss, out = model(sample, training=True) 
                # need to take some measure of closeness of image into accoutn
                image_dif = 1- tf.image.ssim(sample, out, 3)
                loss = internal_loss + image_dif 

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
    X_valid, y_valid = valid_seq.__getitem__(1)

    plt.figure(figsize=(5*8,5))
    n = 8
    idx = 8
    plt.subplot(1, n, 1)
    plt.axis('off') 
    plt.imshow(X_valid[idx],  cmap='gray')
    plt.title("Original")

    model_test = VQVAE()

    # get a list of saved models from file
    saved_models_r = os.listdir(folder)
    saved_models = sorted([".".join(saved_model.split(".")[0:-1]) for saved_model in saved_models_r])[2:]
    saved_models = np.unique(saved_models)
    # saved_models = saved_models[np.where(saved_models=='model_1_1635414388.8481214model_weights')[0][0]:]
    saved_models = saved_models[::30]
    # plot
    for i in range(n-2):
        # model_test.build( input_shape=[10, 256, 256, 1])
        # print(model.summary())
        model_test.load_weights("./saves/initial/"+saved_models[i])
        loss, out1 = model_test(X_valid)
        valid_ssim = (tf.image.ssim(X_valid, out1, 3))
        avg_ssim = np.mean(valid_ssim)
        img_ssim = np.mean((tf.image.ssim(X_valid[idx:idx+1], out1[idx:idx+1], 3)))
        print(avg_ssim, img_ssim)
        plt.subplot(1, n, i+2)
        plt.imshow(out1[idx], cmap='gray')
        plt.axis('off') 
        plt.title("{} iterations \n{:.2f} batch  ssim\n{:.2f} image ssim".format(7*967*(i+2), avg_ssim, img_ssim), fontsize=22)
    plt.savefig('figures/epoch_compare_recon_{}.png'.format(idx))

    # Get encoder/decoder/vq separately for use in pixelcnn
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    vq = model.get_vq()


    # Make data for pixelcnn input
    make_indices_dataset(train_seq, model.get_encoder(), model.get_vq.emb)

    data_dir = pathlib.Path("./")
    codebook_files = list(data_dir.glob('./encoded_data/*'))
    codebook_seq = CodeBookSeq(sorted(codebook_files))

    # make pixel cnn model
    pixel_cnn= PixelCNN()

    # train pixel cnn
    pixel_cnn.compile(optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

    # runs=0
    folder = "./saves/oasis/pixelcnn/"
    model_name="pcnn"

    runs += 1
    num_hundreds = 0
    codebook_indices, _ = codebook_seq.__getitem__(1)
    for i in range(num_hundreds):
        start = time.time()
        pixel_cnn.fit(
            codebook_seq,
        #     batch_size=20,
            epochs=500,
            validation_data=(codebook_indices, codebook_indices),
        )
        time_now = time.time()
        details_1 = "R{}E{}".format(runs, i)
        details_2 = "_{}_".format( time_now) 
        pixel_cnn.save_weights(folder+model_name+details_1+details_2+"model_weights")
        print("Epoch time", time.time()-start)
    
    pixel_cnn.load_weights("./saves/oasis/pixelcnn/pcnnR9E8_1635711787.0700958_model_weights")

    generated = gen_images(pixel_cnn, vq, decoder, codebook_indices.shape[1:], 5)
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(generated_samples[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()


    print("Done")


if __name__ == "__main__":
    main()
