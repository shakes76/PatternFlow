import tensorflow as tf

from modules import *
from dataset import *
import time
import os

"""
Containing the source code for training, validating, testing and saving your model. 
The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”
Make sure to plot the losses and metrics during training.

"""
EPOCHS = 40
BATCH_SIZE = 128
BUFFER_SIZE = 20000
MARGIN = 0.2

MODEL_SAVE_DIR = "E:/ADNI/models"

def siamese_loss(x0, x1, label: int, margin: float) -> float:
    """
    Custom loss function for siamese network.

    Based on contrastive loss.

    Takes two vectors, then calculates their distance.

    Vectors of the same class are rewarded for being close and punished for being far away.
    Vectors of different classes are punished for being close and rewarded for being far away.

    Parameters:
        - x0 -- batch of vectors
        - x1 -- batch of vectors
        - label -- whether or not the two vectors are from the same class. 1 = yes, 0 = no

    Returns:
        - loss value
    """
    dist = tf.reduce_sum(tf.square(x0 - x1), 1)
    dist_sqrt = tf.sqrt(dist)

    loss = label * tf.square(tf.maximum(0., margin - dist_sqrt)) + (1 - label) * dist
    loss = 0.5 * tf.reduce_mean(loss)

    return loss

@tf.function
def train_step(siamese, siamese_optimiser, images1, images2, same_class: bool):
    """
    Executes one step of training the siamese model.
    Backpropogates to update weightings.

    Parameters:
        - siamese -- the siamese network
        - siamese_optimiser -- the optimiser which will be used for backprop
        - images1, images2 -- batch of image data which is either positive or negative
        - same_class -- flag representing whether the two sets of images are of the same class

    Returns:
        - loss value from this the training step
    
    """
    with tf.GradientTape() as siamese_tape:

        x0 = siamese(images1, training=True)
        x1 = siamese(images2, training=True)
        label = int(same_class)

        loss = siamese_loss(x0, x1, label, MARGIN)
    
        siamese_gradients = siamese_tape.gradient(\
            loss, siamese.trainable_variables)

        siamese_optimiser.apply_gradients(zip(
            siamese_gradients, siamese.trainable_variables))

    return loss

def train_siamese_model(model, optimiser, pos_dataset, neg_dataset, epochs):
    start = time.time()
    print("Beginning Siamese Network Training")

    for epoch in range(epochs):
        epoch_start = time.time()

        i = 1
        for pos_batch, neg_batch in zip(pos_dataset, neg_dataset):
            if i % 20 == 0:
              print("-----------------------")
              print("Batch number", i, "complete")
              print(f"{i} batches completed in {time.time() - epoch_start}")
              print(f"Avg batch time: {(time.time() - epoch_start) / i}")

            # alternate between same-same training and same-diff training
            if i % 2 == 0:
                # same training
                same_class = True

                #split batches
                pos1, pos2 = tf.split(pos_batch, num_or_size_splits=2)
                neg1, neg2 = tf.split(neg_batch, num_or_size_splits=2)

                pos_loss = train_step(model, optimiser, pos1, pos2 , same_class)
                neg_loss = train_step(model, optimiser, neg1, neg2 , same_class)
                
            else:
                # diff training
                same_class = False
                diff_loss = train_step(model, optimiser, pos_batch, neg_batch, same_class)

            i += 1
        
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} - training time: {epoch_elapsed}")
    
    elapsed = time.time() - start
    print(f"Siamese Network Training Completed in {elapsed}")

def train_binary_classifier(model, siamese_model, training_data_positive, training_data_negative):
    start = time.time()
    print("Beginning Binary Classifier Training")

    pos_labels = np.ones(training_data_positive.shape[0])
    neg_labels = np.zeros(training_data_negative.shape[0])

    pos_embeddings = siamese_model.predict(training_data_positive)
    neg_embeddings = siamese_model.predict(training_data_negative)

    embeddings = np.concatenate((pos_embeddings, neg_embeddings))
    labels = np.concatenate((pos_labels, neg_labels))

    model.fit(embeddings, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    elapsed = time.time() - start
    print(f"Binary Classifier Training Completed in {elapsed}")

def main():
    # get training data
    training_data_positive = load_data(AD_TRAIN_PATH, "ad_train")
    training_data_negative = load_data(NC_TRAIN_PATH, "nc_train")

    # convert to tensors
    train_data_pos = tf.data.Dataset.from_tensor_slices(training_data_positive
        ).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
    train_data_neg = tf.data.Dataset.from_tensor_slices(training_data_negative
        ).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)

    # build models
    siamese_model = build_siamese()
    binary_classifier = build_binary()
    
    siamese_optimiser = tf.keras.optimizers.Adam(0.05)

    train_siamese_model(siamese_model, siamese_optimiser, train_data_pos, train_data_neg, EPOCHS)
    train_binary_classifier(binary_classifier, siamese_model, training_data_positive, training_data_negative)

    siamese_model.save(os.path.join(MODEL_SAVE_DIR, "siamese_model.h5"))
    binary_classifier.save(os.path.join(MODEL_SAVE_DIR, "binary_model.h5"))

if __name__ == "__main__":
    main()