import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tqdm import tqdm
import tensorflow_addons as tfa
import math
import data as d
import config as c
from perceiver import Perceiver

training_it = d.training_data_iterator()
testing_it = d.test_data_iterator()

def train(model):
    # learning_rate_schedules = tf.keras.optimizers.schedules.CosineDecay(
    #     c.initial_learning_rate, c.num_epochs * c.iterations_per_epoch - c.warm_iterations, alpha=c.minimum_learning_rate, name=None
    # )

    # #SGD optimizer as learnt in class
    # optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)


    # Create LAMB optimizer with weight decay.
    optimizer = tfa.optimizers.LAMB(
        learning_rate=c.learning_rate, weight_decay_rate=c.weight_decay,
    )


    #loss function for train step
    loss_f = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=loss_f,
    )

    #call to initialize
    model(np.zeros((1, 64, 64, 3)))

    model.load_weights("goliath/weights_old.h5")

    for epoch_num in range(c.num_epochs):
        training_it = d.training_data_iterator()
        print(f"epoch:{epoch_num}/{c.num_epochs}")
        total_cor = 0
        total_ce = 0
        for i in tqdm(range(c.iterations_per_epoch)):
            images, labels = training_it.next()
            ce, prediction = train_step(model, images, labels, optimizer, loss_f)
            correct_num = correct_num_batch(labels, prediction)
            print('loss: {:.4f}, accuracy: {:.4f}'.format(ce, correct_num / c.batch_size))
            total_cor += correct_num 
            total_ce += ce

        model.save_weights("weights.h5", save_format='h5')

        with open("train_history.csv", 'a') as f:
            f.write(f"{epoch_num}, {total_cor / c.train_num}, {total_ce / c.train_num}\n")

        print(f"Top-1 (train) OVERALL ACC {total_cor / c.train_num}")

        sum_correct_num = 0
        sum_loss = 0
        testing_it = d.test_data_iterator()
        for i in tqdm(range(c.test_iterations)):
            try:
                images, labels = testing_it.next()
                loss, prediction = test_step(model, images, labels, loss_f)
                # print("labels, pred", labels, prediction)
                correct_num = correct_num_batch(labels, prediction)
                print(f"correct num:{correct_num}")
                sum_correct_num += correct_num 
                sum_loss += loss
                print('loss: {:.4f}, accuracy: {:.4f}'.format(loss, correct_num / c.batch_size))
            except Exception as e:
                print(e)
                print("probs ran out, continuing")
                
            print(f"TEST {sum_correct_num / c.test_num}, {sum_loss / c.test_num}\n")


    

        with open("test_history.csv", 'a') as f:
            f.write(f"{epoch_num}, {sum_correct_num / c.test_num}, {sum_loss / c.test_num}\n")  

# perceiver_classifier = Perceiver(
#     c.patch_size,
#     c.num_patches,
#     c.latent_dim,
#     c.projection_dim,
#     c.num_heads,
#     c.num_transformer_blocks,
#     c.ffn_units,
#     c.dropout_rate,
#     c.num_iterations,
#     c.classifier_units,
# )
perceiver_classifier = Perceiver()

@tf.function
def train_step(model, images, labels, optimizer, loss_f):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss = loss_f(labels, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def test_step(model, images, labels, loss_f):
    prediction = model(images, training=False)
    loss = loss_f(labels, prediction)
    return loss, prediction

def correct_num_batch(y_true, y_pred):
    pred = tf.argmax(y_pred, -1)
    pred = tf.reshape(pred, (min(c.batch_size, pred.shape[0]), 1))
    pred = tf.cast(pred, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)

    correct_num = tf.equal(y_true, pred)
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num

train(perceiver_classifier)