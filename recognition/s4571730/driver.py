import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from model import Perceiver
import random, os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Constants
# IMAGE_DIR = 'AKOA_Analysis'
IMAGE_DIR = 'D:/AKOA_Analysis_orig/'
BATCH_SIZE = 32
IMG_SIZE = (73, 64) # image resize
ROWS, COLS = IMG_SIZE
TEST_PORTION = 5 # portion of validation set to become test set
SHUFFLE_RATE = 512
AUTO_TUNE = tf.data.experimental.AUTOTUNE

LATENT_SIZE = 256  # Size of the latent array.
NUM_BANDS = 6 # Number of bands in Fourier encode. Used in the paper
NUM_CLASS = 1 # Number of classes to be predicted (1 for binary)
PROJ_SIZE = 2*(2*NUM_BANDS + 1) + 1  # Projection size of data after fourier encoding
NUM_HEADS = 8  # Number of Transformer heads.
NUM_TRANS_BLOCKS = 6 # Number of transformer blocks in the transformer layer. Used in the paper
NUM_ITER = 8  # Repetitions of the cross-attention and Transformer modules. Used in the paper
MAX_FREQ = 10 # Max frequency in Fourier encode
LR = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 10
TRAIN_SPLIT = 0.8

"""
Create train, validate and test dataset
Images have to be in left and right folders 

AKOA_Analysis/
...left/
......left_image_1.jpg
......left_image_2.jpg
...right/
......right_image_1.jpg
......right_image_2.jpg

"""

def proof_no_set_overlap(train_image_names, test_image_names):
    """
    A method for proving the training and validation sets have no overlapping
    patients, and hence no data-leakage.
    Args:
        train_image_names: a list of names of images in training set
        test_image_names: a list of names of images in validation set
    """
    unique_train_patients = []
    unique_test_patients = []

    for train_name in train_image_names:
        if train_name[:10] not in unique_train_patients:
            unique_train_patients.append(train_name[:10])
    for test_name in test_image_names:
        if test_name[:10] not in unique_test_patients:
            unique_test_patients.append(test_name[:10])

    print("unique patients in training set: ", len(unique_train_patients))
    print("unique patients in testing set: ", len(unique_test_patients))

    matches = len([x for x in unique_train_patients
                   if x in unique_test_patients])

    print("number of unique patients in both training and testing: ", matches)


def split_by_patients(image_names, train_split):
    """
    A method for splitting the dataset into training and testing sets.
    The returned sets have no data leakage by ensuring that a patient is unique
    to only 1 of the sets. This is verified by proof_no_set_overlap().
    Args:
        image_names: A list of all the image names in the full dataset
        N_train: The number of images to have in the training set
        N_test: The number of images to have in the validation set

    Returns:
        training_image_names: A list of image names apart of training set
        testing_image_names: A list of image names apart of validation set
    """
    patient_batches = dict()
    train_image_names = []
    test_image_names = []

    for name in image_names:
        patient_id = name.split('_')[0]
        if patient_id in patient_batches:
            patient_batches[patient_id].append(name)
        else:
            patient_batches[patient_id] = [name]
    print("unique patients in entire dataset: ", len(patient_batches))

    building_train = True
    for patient_batch in patient_batches.values():
        for name in patient_batch:
            if building_train:  # first step: building training set
                if len(train_image_names) <= len(image_names) * train_split:
                    train_image_names.append(name)
                else:
                    building_train = False  # start building test set now
                    break
            else:  # second step: building testing set
                if len(test_image_names) <= len(image_names) * (1-train_split):
                    test_image_names.append(name)
                else:
                    break  # done building test set

    return train_image_names, test_image_names

"""
Helper function for loading a X and y set based of the image names
Params:
    image_names: The image names to build the data set from

Returns:
    X_set, y_set: numpy array of X (image data) and y (labels)
"""
def get_data(dir, image_names):
    X_set = []
    y_set = []
    for file_name in image_names:
        # Greyscale is used since the images are black and white only
        image = load_img(dir + file_name,
                         target_size=IMG_SIZE,
                         color_mode="grayscale")

        # convert an image to numpy array
        X_set.append(img_to_array(image))

        # Determine left or right label based on dataset filename
        # left: 0, right: 1
        y_set.append(1) if \
            "RIGHT" in file_name or "R_I_G_H_T" in file_name or \
            "Right" in file_name or "right" in file_name \
        else y_set.append(0)
            

    X_set = np.array(X_set)
    X_set /= 255.0
    y_set = np.array(y_set).flatten()
    return X_set, y_set

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

    # IDs in train and test set, for overlap checking
    train_ids = set()
    test_ids = set()

    X_train, y_train, X_test, y_test = [], [], [], []

    for file_name in all_files:
        # ID is OAIxxxxxxxx, ends before the first _
        patient_id = file_name.split('_')[0]
        if patient_id in patient_id_to_files:
            patient_id_to_files[patient_id].append(file_name) 
        else:
            patient_id_to_files[patient_id] = [file_name]
            
    # Lambda function to determine label based on filename
    # left: 0, right: 1
    label = lambda file_name: 1 if \
        "RIGHT" in file_name or "R_I_G_H_T" in file_name or \
        "Right" in file_name or "right" in file_name \
            else 0

    # Lambda function to load an image in greyscale mode
    img_load = lambda file_name: load_img(dir + file_name,
                         target_size=IMG_SIZE,
                         color_mode="grayscale")

    # Loop each group of files belonging to a patient
    change_to_test = False
    for patient_id, patient_files in patient_id_to_files.items():
        # Loop each file in that group
        train_ids.add(patient_id) if not change_to_test else test_ids.add(patient_id)
        for file_name in patient_files:
            if not change_to_test:
                if len(X_train) <= len(all_files) * train_split:
                    img = img_to_array(img_load(file_name))
                    X_train.append(img)
                    y_train.append(label(file_name))
                else:
                    change_to_test = True
                    break
            else:
                img = img_to_array(img_load(file_name))
                X_test.append(img)
                y_test.append(label(file_name))
                

    # Proof that train ids and test ids dont overlap
    print("Unique patients in dataset: ", len(patient_id_to_files))
    print("Unique patients in train ds: ", len(train_ids))
    print("Unique patients in test ds: ", len(test_ids))
    print("Overlap: ", train_ids.intersection(test_ids))

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
    X_val, y_val = X_test[0:len(X_test) // TEST_PORTION * (TEST_PORTION - 1)], \
        y_test[0:len(y_test) // TEST_PORTION * (TEST_PORTION - 1)]

    X_test, y_test = X_val[len(X_val) // TEST_PORTION * (TEST_PORTION - 1):], \
        y_test[len(y_test) // TEST_PORTION * (TEST_PORTION - 1):]

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataset(image_dir, img_size):
    # Training dataset, shuffle is True by default
    training_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                    subset="training", seed=46, label_mode='int',
                                                    batch_size=1, image_size=img_size)
    # Validation dataset
    validation_dataset = image_dataset_from_directory(image_dir, validation_split=0.2, color_mode='grayscale',
                                                      subset="validation", seed=46, label_mode='int',
                                                      batch_size=1, image_size=img_size)

    # Test dataset, taking 1/5 of the validation set
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // TEST_PORTION)
    validation_dataset = validation_dataset.skip(val_batches // TEST_PORTION)

    # normalize and prefetch images for faster training
    training_dataset = training_dataset.map(process).prefetch(AUTO_TUNE)
    validation_dataset = validation_dataset.map(process).prefetch(AUTO_TUNE)
    test_dataset = test_dataset.map(process).prefetch(AUTO_TUNE)

    return training_dataset, validation_dataset, test_dataset

"""
Convert dataset object to numpy array
"""
def dataset_to_numpy_util(dataset, len_ds):
  dataset = dataset.batch(len_ds)
  for images, labels in dataset:
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    break

  return numpy_images, numpy_labels

"""
Normalize image to range [0,1]
"""
def process(image,label):
    image = tf.cast(image / 255. ,tf.float32)
    return image,label

def get_numpy_ds():
    training_set, validation_set, test_set = create_dataset(IMAGE_DIR, IMG_SIZE)
    X_train, y_train = dataset_to_numpy_util(training_set, len(training_set))
    X_train = X_train.reshape((len(training_set), ROWS, COLS, 1))

    X_val, y_val = dataset_to_numpy_util(validation_set, len(validation_set))
    X_val = X_val.reshape((len(validation_set), ROWS, COLS, 1))

    X_test, y_test = dataset_to_numpy_util(test_set, len(test_set))
    X_test = X_test.reshape((len(test_set), ROWS, COLS, 1))
    del training_set
    del validation_set
    del test_set
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def plot_data(history):
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
    plt.show()

if __name__ == "__main__":

    SAVE_DATA = True
    if SAVE_DATA:
        X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, TRAIN_SPLIT)
        np.save("D:/np/X_train.npy", X_train)
        np.save("D:/np/y_train.npy", y_train)
        np.save("D:/np/X_val.npy", X_val)
        np.save("D:/np/y_val.npy", y_val)
        np.save("D:/np/X_test.npy", X_test)
        np.save("D:/np/y_test.npy", y_test)

    else:
        X_train = np.load("D:/np/X_train.npy")
        y_train = np.load("D:/np/y_train.npy")
        X_val = np.load("D:/np/X_val.npy")
        y_val = np.load("D:/np/y_val.npy")
        X_test = np.load("D:/np/X_test.npy")
        y_test = np.load("D:/np/y_test.npy")

    # Initialize the model
    # knee_model = Perceiver(patch_size=0,
    #                         data_size=ROWS*COLS, 
    #                         latent_size=LATENT_SIZE,
    #                         num_bands=NUM_BANDS,
    #                         proj_size=PROJ_SIZE, 
    #                         num_heads=NUM_HEADS,
    #                         num_trans_blocks=NUM_TRANS_BLOCKS,
    #                         num_iterations=NUM_ITER,
    #                         max_freq=MAX_FREQ,
    #                         lr=LR,
    #                         weight_decay=WEIGHT_DECAY,
    #                         epoch=EPOCHS)


    # checkpoint_dir = './ckpts'
    # checkpoint = tf.train.Checkpoint(
    #         knee_model=knee_model)
    # ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # # checkpoint.restore(ckpt_manager.latest_checkpoint)
    # history = knee_model.train(
    #                 train_set=(X_train, y_train),
    #                 val_set=(X_val, y_val),
    #                 test_set=(X_test, y_test),
    #                 batch_size=BATCH_SIZE)

    # ckpt_manager.save()
    # plot_data(history)

    # # Retrieve a batch of images from the test set
    # image_batch, label_batch = X_test[:BATCH_SIZE], y_test[:BATCH_SIZE]
    # image_batch = image_batch.reshape((BATCH_SIZE, ROWS, COLS, 1))
    # predictions = knee_model.predict_on_batch(image_batch).flatten()
    # label_batch = label_batch.flatten()

    # # Fix the preds into 0 (left) or 1 (right)
    # predictions = tf.where(predictions < 0.5, 0, 1).numpy()
    # class_names = {0: "left", 1: "right"}

    # # Plot preds
    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image_batch[i], cmap="gray")
    #     plt.title("pred: " + class_names[predictions[i]] + ", real: " + class_names[label_batch[i]])
    #     plt.axis("off")
    # plt.show()