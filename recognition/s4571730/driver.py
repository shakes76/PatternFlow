import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from model import train, Perceiver
import random, os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Constants
# IMAGE_DIR = 'AKOA_Analysis'
IMAGE_DIR = 'D:/AKOA_Analysis_orig'
BATCH_SIZE = 32
IMG_SIZE = (64, 64) # image resize
ROWS, COLS = IMG_SIZE
TEST_PORTION = 5 # portion of validation set to become test set
SHUFFLE_RATE = 512
AUTO_TUNE = tf.data.experimental.AUTOTUNE

LATENT_SIZE = 256  # Size of the latent array.
NUM_BANDS = 4 # Number of bands in Fourier encode. Used in the paper
NUM_CLASS = 1 # Number of classes to be predicted (1 for binary)
PROJ_SIZE = 2*(2*NUM_BANDS + 1) + 1  # Projection size of data after fourier encoding
NUM_HEADS = 8  # Number of Transformer heads.
NUM_TRANS_BLOCKS = 6 # Number of transformer blocks in the transformer layer. Used in the paper
NUM_ITER = 8  # Repetitions of the cross-attention and Transformer modules. Used in the paper
MAX_FREQ = 10 # Max frequency in Fourier encode
LR = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 10

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


def process_dataset(dir_data, train_split):
    """
    A function for creating the tf arrays for the X and y training and
    validation sets from the image files in this directory 'dir_data'.
    Ensures no data leakage in sets by calling split_by_patients()
    Args:
        dir_data: A directory where all images in the dataset are located
        N_train: The number of images to have in the training set
        N_test: The number of images to have in the validation set

    Returns:
        X_train, y_train: The tf array formatted images and their labels
         for the training set
        X_test, y_test: The tf array formatted images and their labels
         for the validation set
    """
    all_image_names = os.listdir(dir_data)
    # num_total: 18680
    # num left found: 7,760
    # num_right found: 10,920

    train_image_names, test_image_names = split_by_patients(all_image_names,
                                                            train_split)

    random.shuffle(train_image_names)
    random.shuffle(test_image_names)

    proof_no_set_overlap(train_image_names, test_image_names)

    img_shape = (64, 64)

    def get_data(image_names):
        """
        Helper function for loading a X and y set based of the image names
        Args:
            image_names: The image names to build the data set from

        Returns:
            X_set, y_set: the tf array of the X and y set built
        """
        X_set = []
        y_set = []
        for i, name in enumerate(image_names):
            image = load_img(dir_data + "/" + name,
                             target_size=(img_shape), color_mode="grayscale")

            # normalise image pixels
            image = img_to_array(image)

            X_set.append(image)
            if "LEFT" in name or "L_E_F_T" in name or \
                    "Left" in name or "left" in name:
                label = 0
            else:
                label = 1

            y_set.append(label)

        X_set = np.array(X_set)
        X_set /= 255.0

        return X_set, np.array(y_set, dtype=np.uint8).flatten()

    X_train, y_train = get_data(train_image_names)
    X_val, y_val = get_data(test_image_names)
    X_test, y_test = X_val[len(X_val) // 5 * 4:], y_val[len(y_val) // 5 * 4:]
    X_val, y_val = X_val[0:len(X_val) // 5 * 4], y_val[0:len(y_val) // 5 * 4]

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

    # generate dataset. Convert to numpy array for eaiser batch size tracking (needed in fourier encode)
    # X_train, y_train, X_val, y_val, X_test, y_test = get_numpy_ds()
    # img_num = 14944
    train_split = 0.8
    X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, train_split)
    print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test))
    print(X_train.shape)
    # # Initialize the model
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
    # history = train(knee_model,
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

    # predictions = tf.where(predictions < 0.5, 0, 1).numpy()
    # class_names = {0: "left", 1: "right"}

    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image_batch[i], cmap="gray")
    #     plt.title("pred: " + class_names[predictions[i]] + ", real: " + class_names[label_batch[i]])
    #     plt.axis("off")
    # plt.show()


