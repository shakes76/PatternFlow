import tensorflow as tf

from modules import *
from dataset import *
import time
import os
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

EPOCHS = 100
BATCH_SIZE = 64
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
        - x0, x1 -- tensor batch of vectors. Shape: (batch size, embedding size)
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
            shape: (batch size, width, height, number of channels)
        - same_class -- bool flag representing whether the two sets of images are of the same class

    Returns:
        - loss value from this the training step
    
    """
    with tf.GradientTape() as siamese_tape:

        # convert images to embeddings
        x0 = siamese(images1, training=True)
        x1 = siamese(images2, training=True)
        label = int(same_class)

        loss = siamese_loss(x0, x1, label, MARGIN)
    
        siamese_gradients = siamese_tape.gradient(\
            loss, siamese.trainable_variables)

        siamese_optimiser.apply_gradients(zip(
            siamese_gradients, siamese.trainable_variables))

    return loss


def train_siamese_model(model, optimiser, pos_dataset, neg_dataset, epochs) -> None:
    """
    Trains the siamese model.

    Alternates between training images of the same class then images of different classes.

    Parameters:
        - model -- the siamese model to train
        - optimiser -- the optimiser used for back propogation
        - pos_dataset, neg_dataset -- pre-batched tensorflow dataset
        - epochs -- number of epochs to train for
    """

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


def train_binary_classifier(model, siamese_model, training_data_positive, training_data_negative) -> None:
    """
    Trains the binary classifier used to classify the images into one of the two classes.

    Converts raw data to embeddings then fits the model.

    Parameters:
        - model -- the binary classification model to train
        - siamese_model -- the pre-trained siamese model used to generate embeddings
        - training_data_positive, training_data_negative -- raw image data
    """
    start = time.time()
    print("Beginning Binary Classifier Training")

    # generate labels - 1: positive, 0: negative
    pos_labels = np.ones(training_data_positive.shape[0])
    neg_labels = np.zeros(training_data_negative.shape[0])

    # convert image data to embeddings
    pos_embeddings = siamese_model.predict(training_data_positive)
    neg_embeddings = siamese_model.predict(training_data_negative)

    # merge positive and negative datasets
    embeddings = np.concatenate((pos_embeddings, neg_embeddings))
    labels = np.concatenate((pos_labels, neg_labels))

    history = model.fit(embeddings, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    elapsed = time.time() - start
    print(f"Binary Classifier Training Completed in {elapsed}")

    return history

def run_pca(siamese_model, training_data_positive, training_data_negative):
    """
    Run Principle Component Analysis on the Siamese Model Embeddings and plot the
     two features with the highest variance.

    Code adapted from https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

    Parameters:
        - siamese_model -- The model with which to generate the embeddings to perform PCA on
        - training_data_positive, training_data_negative -- raw image data as numpy arrays
    """
    
    pos_labels = np.ones(training_data_positive.shape[0])
    neg_labels = np.zeros(training_data_negative.shape[0])

    pos_embeddings = siamese_model.predict(training_data_positive)
    neg_embeddings = siamese_model.predict(training_data_negative)

    embeddings = np.concatenate((pos_embeddings, neg_embeddings))
    labels = np.concatenate((pos_labels, neg_labels))

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)

    principalDf = pd.DataFrame(data = principal_components
             , columns = ['principal component 1', 'principal component 2'])
    labelsDf = pd.DataFrame(labels, columns=["label"])
    finalDf = pd.concat([principalDf, labelsDf], axis = 1)

    # plot first two principal components
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1.0, 0.0]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(targets)
    ax.grid()

def main():
    """
    Trains the models

    Loads training data using dataset.py
    Generates the models using modules.py
    Uses functions defined above to train the models
    Saves the models for later prediction
    """

    # get training data
    training_data_positive = load_data(AD_TRAIN_PATH, "ad_train")
    training_data_negative = load_data(NC_TRAIN_PATH, "nc_train")

    # convert to tensors for siamese training
    train_data_pos = tf.data.Dataset.from_tensor_slices(training_data_positive
        ).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
    train_data_neg = tf.data.Dataset.from_tensor_slices(training_data_negative
        ).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)

    # build models
    siamese_model = build_siamese()
    binary_classifier = build_binary()
    
    # create optimiser for siamese model
    siamese_optimiser = tf.keras.optimizers.Adam(0.05)

    # train the models
    train_siamese_model(siamese_model, siamese_optimiser, train_data_pos, train_data_neg, EPOCHS)

    # optionally, run principle component analysis on siamese model to assess embeddings
    run_pca(siamese_model, training_data_positive, training_data_negative)

    train_binary_classifier(binary_classifier, siamese_model, training_data_positive, training_data_negative)

    # save the models
    siamese_model.save(os.path.join(MODEL_SAVE_DIR, "siamese_model.h5"))
    binary_classifier.save(os.path.join(MODEL_SAVE_DIR, "binary_model.h5"))

    

if __name__ == "__main__":
    main()