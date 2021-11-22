import tensorflow as tf
import matplotlib.pyplot as plt
from model import Perceiver
import random, os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Constants
IMAGE_DIR = 'D:/AKOA_Analysis/' # Path to image dataset
BATCH_SIZE = 32 # Batch size in traning
IMG_SIZE = (73, 64) # image resize
ROWS, COLS = IMG_SIZE
TEST_PORTION = 5 # 1/n of test set to become real test set (the rest becomes validation)
LATENT_SIZE = 256  # Size of the latent array.
NUM_BANDS = 6 # Number of bands in Fourier encode. Used in the paper
NUM_CLASS = 1 # Number of classes to be predicted (1 for binary)
PROJ_SIZE = 2*(2*NUM_BANDS + 1) + 1  # Projection size of data after fourier encoding
NUM_HEADS = 8  # Number of Transformer heads.
NUM_TRANS_BLOCKS = 6 # Number of transformer blocks in the transformer layer. Used in the paper
NUM_ITER = 8  # Repetitions of the cross-attention and Transformer modules. Used in the paper
MAX_FREQ = 10 # Max frequency in Fourier encode
LR = 0.001 # Learning rate for optimizer
WEIGHT_DECAY = 0.0001 # Weight decay for optimizer
EPOCHS = 10 # Number of epochs in training
TRAIN_SPLIT = 0.8 # Portion for training set

"""
Shuffle the train and test ds, then split test ds into validation and test

Params:
    X_train: train ds
    y_train: train labels
    X_test: test ds
    y_test: test labels

Returns: a tuple of train, val and test ds, with labels
"""
def split_and_shuffle(X_train, y_train, X_test, y_test):
    # Shuffle the dataset
    indices_train = list(range(0, len(X_train)))
    indices_test = list(range(0, len(X_test)))

    random.shuffle(indices_train)
    random.shuffle(indices_test)

    X_train = np.array(X_train)
    X_train /= 255.0
    X_train = X_train[indices_train]

    y_train = np.array(y_train)
    y_train = y_train[indices_train]

    X_test = np.array(X_test)
    X_test /= 255.0
    X_test = X_test[indices_test]

    y_test = np.array(y_test)
    y_test = y_test[indices_test]

    # split the test set into validation and test, with 1/TEST_PORTION values in test set
    len_val = len(X_test) // TEST_PORTION * (TEST_PORTION - 1)
    X_val, y_val = X_test[0:len_val], y_test[0:len_val]
    X_test, y_test = X_test[len_val:], y_test[len_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


"""
Verify that no train patient ids get into test patient ids

Params:
    patient_ids: Dict mapping id to a list of files
    train_ids: Set containing train patient ids
    test_ids: Set containing test patient ids

Returns: 
    nothing, but assert that the intersection between train and test patients is 0
"""
def verify_no_leakage(patient_ids, train_ids, test_ids):
    # Proof that train ids and test ids dont overlap
    print("Unique patients in dataset: ", len(patient_ids))
    print("Unique patients in train ds: ", len(train_ids))
    print("Unique patients in test ds: ", len(test_ids))
    print("Overlap: ", train_ids.intersection(test_ids))

    assert len(train_ids.intersection(test_ids)) == 0

"""
Create numpy array of images data and labels, using the supplied directory path.
Ensure no data leakage in the process. 

Params:
    dir_data: A directory where all images in the dataset are located
    train_split: Split ratio of training dataset (the remaining is for
    validation and test)

Returns:
    A tuple of training, validation and testing dataset, with their labels.
"""
def process_dataset(dir, train_split):

    # List of all file names in the directory
    all_files = os.listdir(dir)

    # Map patient ID to their files
    patient_id_to_files = dict()
    # Load file names to respective patients
    for file_name in all_files:
        # ID is OAIxxxxxxxx, ends before the first _
        patient_id = file_name.split('_')[0]
        if patient_id in patient_id_to_files:
            patient_id_to_files[patient_id].append(file_name) 
        else:
            patient_id_to_files[patient_id] = [file_name]

    # Shuffle the dict for better training    
    shuffled = list(patient_id_to_files.items())
    random.shuffle(shuffled)
    patient_id_to_files = dict(shuffled)

    # Lambda function to determine label based on filename
    # left: 0, right: 1
    label = lambda file_name: 1 if \
        "RIGHT" in file_name or "R_I_G_H_T" in file_name or \
        "Right" in file_name or "right" in file_name \
            else 0

    # Lambda function to load an image in greyscale mode
    load_image = lambda file_name: img_to_array(load_img(dir + file_name,
                         target_size=IMG_SIZE,
                         color_mode="grayscale"))

    # Flag marking when to switch to test set
    change_to_test = False

    X_train, y_train, X_test, y_test = [], [], [], []
    # IDs in train and test set, for overlap checking
    train_ids = set()
    test_ids = set()

    # Loop each group of files belonging to a patient
    for patient_id, patient_files in patient_id_to_files.items():
        # Loop each file in that group
        train_ids.add(patient_id) if not change_to_test else test_ids.add(patient_id)
        for file_name in patient_files:
            if not change_to_test:
                if len(X_train) <= len(all_files) * train_split:
                    X_train.append(load_image(file_name))
                    y_train.append(label(file_name))
                else:
                    change_to_test = True
                    # Break ensures moving to the next patient
                    break
            else:
                X_test.append(load_image(file_name))
                y_test.append(label(file_name))
    
    verify_no_leakage(patient_id_to_files, train_ids, test_ids)

    # Shuffle the ds, split test into test and val and return
    return split_and_shuffle(X_train, y_train, X_test, y_test)


"""
Plot the training history

Params:
    history: history generated by model.fit()

Returns:
    nothing, but plots training acc and loss
"""
def plot_history(history):
    # Plotting the Learning curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.show()


"""
Plot some predictions of the model and compare with actual labels

Params:
    X_test: test dataset images
    y_test: test dataset labels
    
Returns:
    nothing, but plots the predictons
"""
def visualize_preds(knee_model, X_test, y_test):
    class_names = {0: "left", 1: "right"}
    # Retrieve a batch of images from the test set
    image_batch, label_batch = X_test[:BATCH_SIZE], y_test[:BATCH_SIZE]
    image_batch = image_batch.reshape((BATCH_SIZE, ROWS, COLS, 1)) # 1D fpr greyscale image
    predictions = knee_model.predict_on_batch(image_batch).flatten()
    label_batch = label_batch.flatten()

    # Fix the preds into 0 (left) or 1 (right)
    predictions = tf.where(predictions < 0.5, 0, 1).numpy()

    # Plot predictions
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i], cmap="gray")
        plt.title("pred: " + class_names[predictions[i]] + ", real: " + class_names[label_batch[i]])
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # True if need togenerate and save data from the dataset
    # False if only need to load processed data from disk
    SAVE_DATA = False
    if SAVE_DATA:
        X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, TRAIN_SPLIT)
        np.save("D:/np/X_train.npy", X_train)
        np.save("D:/np/y_train.npy", y_train)
        np.save("D:/np/X_val.npy", X_val)
        np.save("D:/np/y_val.npy", y_val)
        np.save("D:/np/X_test.npy", X_test)
        np.save("D:/np/y_test.npy", y_test)

        print(len(X_train), len(X_test), len(X_val), len(y_train), len(y_test), len(y_val))

    else:
        X_train = np.load("D:/np/X_train.npy")
        y_train = np.load("D:/np/y_train.npy")
        X_val = np.load("D:/np/X_val.npy")
        y_val = np.load("D:/np/y_val.npy")
        X_test = np.load("D:/np/X_test.npy")
        y_test = np.load("D:/np/y_test.npy")

        print(len(X_train), len(X_test), len(X_val), len(y_train), len(y_test), len(y_val))

    class_names = {0: "left", 1: "right"}

    # Visualize some images of the dataset
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(X_train[i], cmap="gray")
    #     plt.title(class_names[y_train[i]])
    #     plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    # Initialize the model
    knee_model = Perceiver(data_size=ROWS*COLS, 
                           latent_size=LATENT_SIZE,
                           num_bands=NUM_BANDS,
                           proj_size=PROJ_SIZE, 
                           num_heads=NUM_HEADS,
                           num_trans_blocks=NUM_TRANS_BLOCKS,
                           num_iterations=NUM_ITER,
                           max_freq=MAX_FREQ,
                           lr=LR,
                           weight_decay=WEIGHT_DECAY,
                           epoch=EPOCHS)
                           

    # Checkpoint for saving the model
    checkpoint_dir = './ckpts'
    checkpoint = tf.train.Checkpoint(
            knee_model=knee_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # checkpoint.restore(ckpt_manager.latest_checkpoint)
    history = knee_model.train(
                    train_set=(X_train, y_train),
                    val_set=(X_val, y_val),
                    test_set=(X_test, y_test),
                    batch_size=BATCH_SIZE)

    ckpt_manager.save()
    plot_history(history)
    visualize_preds(knee_model, X_test, y_test)

