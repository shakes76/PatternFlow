import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import dataset
import modules


def main():
    AD_dataset = dataset.load_data('./AD_NC/train/AD', (224, 224))
    NC_dataset = dataset.load_data('./AD_NC/train/NC', (224, 224))

    pos_pair1, pos_pair2, neg_pair1, neg_pair2 = dataset.make_pair(AD_dataset, NC_dataset)

    choice_dataset = dataset.shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2)

    train_dataset, validation_dataset = dataset.split_dataset(choice_dataset, 16, 100)

    # for img1, img2, label in train_dataset.take(1):
    #     dataset.visualize(img1, img2, label)

    # create the model
    siamese = modules.siamese


    if training:
        optimizer = tf.keras.optimizers.Adam(0.00006)
        loss_tracker = tf.keras.metrics.Mean(name="loss")
        acc_tracker = tf.keras.metrics.Mean(name="accuracy")

        # save the accuracy and loss for each epoch
        train_loss_results = []
        train_accuracy_results = []


        for epoch in range(1, 11):
            print("Start of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            for step, (x1, x2, y) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Compute the output of the model
                    logits = siamese([x1, x2], training=True)

                    # expand the labels to match the shape of logits
                    y = tf.expand_dims(y, 1)

                    # Compute the loss value
                    loss = tf.keras.losses.binary_crossentropy(y, logits)
                    # Compute the accuracy 
                    acc = tf.keras.metrics.binary_accuracy(y, logits)

                    # Compute the gradient of the loss with respect to the parameters of the model
                    grads = tape.gradient(loss, siamese.trainable_weights)
                    # Update the weights of the model
                    optimizer.apply_gradients(zip(grads, siamese.trainable_weights))

                    # update loss and accuracy metrics
                    loss_tracker.update_state(loss)
                    acc_tracker.update_state(acc)

                if step % 100 == 0:
                    print("loss: ", loss_tracker.result().numpy(), "accuracy: ", acc_tracker.result().numpy())


            # save the loss and accuracy for each epoch
            train_loss_results.append(loss_tracker.result().numpy())
            train_accuracy_results.append(acc_tracker.result().numpy())

            # save the weights of the model after every epoch as checkpoints
            checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
            siamese.save_weights(checkpoint_path.format(epoch=epoch))
    else:
        # load the weights of the model
        siamese.load_weights(tf.train.latest_checkpoint("checkpoints"))

    for img1, img2, label in train_dataset.take(1):
        dataset.visualize(img1, img2, label, to_show=4, num_col=4, predictions=siamese([img1, img2], training=False).numpy(), test=True)

if __name__ == '__main__':
    main()