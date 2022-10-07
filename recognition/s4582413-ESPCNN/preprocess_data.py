import os # for file processing and reading
from random import random # for data shuffling
import numpy as np # for linear algebra
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from Perceiver import Perceiver

IMG_DIR = "" #Some directory that contains the image
IMG_SIZE = (73, 64)

# Maps the id to file respectively and returns the file
def id_to_files(files, id_to_files):
    file_check = files[0].split("_")[0]
    print(file_check)
    # we see how files are in format of OAI..._otherstuff
    # so file.split("_")[0] represents patient id(pid)

    id_map = dict()  # map is empty from start

    for file in files:
        if file.split("_")[0] in id_to_files:
            id_map[file.split("_")[0]].append(file)
        else:
            id_to_files[file.split("_")[0]] = [files]

    return id_to_files


# Maps the file to its respective label and returns it, with 1 representing right and 0 being left
def get_label(file):
    label = 0 # label of the file, 0 being left and 1 being right
    file = file.replace("_", "") # replace redundant symbols in the filename
    if "RIGHT" in file or "Right" in file or "right" in file:
        label = 1
    else:
        label = 0

    return label


# Gets the image the of the file represented in array format (with color mode being gray scale) 
def get_img(directory, img_size):
    img = img_to_array(load_img(directory, target_size=img_size, color_mode="grayscale"))

    return img


# sanity check for intersection between test and training id set, otherwise we would artificially get 
# high accuracy for test set data
def intersection_check(train_pids, test_pids):
    # display the unique id in training and testing procedure
    print("unique pid in train: ", len(train_pids))
    print("unique pid in test: ", len(test_pids))

    # sanity check
    assert (len(train_pids.intersection(test_pids)) == 0)


# Splits the data set randomly now to add in a validation set within them
# and returns the training, test and validation data set
def random_split(X_train, X_test, y_train, y_test):
    # randomnly shuffling the data set
    train_indices = list(range(0, len(X_train)))
    test_indices = list(range(0, len(y_train)))
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Normalising the data set as well as applying the shuffling to it
    X_train = np.array(X_train)
    X_train /= 255.0
    X_train = X_train[train_indices]
    y_train = np.array(y_train)
    y_train = y_train[train_indices]

    X_test = np.array(X_test)
    X_test /= 255.0
    X_test = X_test[test_indices]
    y_test = np.array(y_test)
    y_test = y_test[test_indices]

    # Splt some of the test set as validation set
    split_len = len(X_test) // 2 # want half test set as validation while remaining half to be test set
    X_test = X_test[0:split_len]
    y_test = y_test[0:split_len]
    X_val = X_test[split_len:]
    y_val = y_test[split_len:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Processes the data set, given the directory of the images
# returns the training, validation and test data set
def process_dataset(dir):
    files = os.listdir(dir)

    # First obtain mapping from patient id to files, so can identify what files each id map to
    pid_file_map = dict() # mapping from patient id (pid) to its corresponding file
    pid_file_map = id_to_files(files, pid_file_map) # Obtain the mapping from pid to files

    # Now create the train and test set
    # Note that we are either training the model or testing the model, so only
    # X_train, y_train will be used or X_test, y_test will be used
    # by default is false as we assume we are training the data
    being_tested = False
    X_train, y_train = [], []
    X_test, y_test = [], []

    # The pids the training and test set, this is to ensure that there are no intersection of patient id between
    # training and test set (otherwise the testing wouldn't reflect on the model's performance)
    train_pids = set()
    test_pids = set()

    for pid, pid_files in pid_file_map.items():
        if not being_tested:
            train_pids.add(pid)
        else:
            test_pids.add(pid)
        # loop through all the files for that patient
        for file in pid_files:
            if not being_tested:
                exceed_size = len(X_train) > len(files) * 0.8
                if not exceed_size:
                    X_train.append(get_img(dir+file, IMG_SIZE)) # the image represented in array format
                    y_train.append(get_label(file)) # the label for the file
                else:
                    being_tested = True # now handling test cases
                    break # exit the most inner for-loop
            else:
                X_test.append(get_img(dir+file, IMG_SIZE))
                y_test.append(get_label(file))

    intersection_check(train_pids, test_pids)

    return random_split(X_train, X_test, y_train, y_test)


"""
Plots the training history to spot the learning curve of the model
"""
def plot_learning_curve(training_history):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(training_history.history['acc'], label='Training Accuracy')
    plt.plot(training_history.history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel("Epoch num")
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Accuracy on training and validation data set')
    plt.subplot(2, 1, 2)
    plt.plot(training_history.history['loss'], label='Training Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch num")
    plt.ylabel('Cross Entropy loss')
    plt.ylim([0, 1.0])
    plt.title('Loss on training and validation data set')
    plt.xlabel('Epoch num')
    plt.tight_layout()
    plt.show()

"""
Plots some prediction of the model with the images true labels
"""
def plot_prediction(model, test_data):
    label_map = {0: "left", 1: "right"}
    # Retrieve a batch of images from the test set
    images, labels = test_data[0][:BATCH_SIZE], test_data[1][:BATCH_SIZE]
    images = images.reshape((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))  # 1D fpr greyscale image
    predictions = model.predict_on_batch(images).flatten()
    labels = labels.flatten()
    # Fix the preds into 0 (left) or 1 (right)
    predictions = tf.where(predictions < 0.5, 0, 1).numpy()
    # Plot predictions
    plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title("pred: " + label_map[predictions[i]] + ", real: " + label_map[labels[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def main():
    # Indicating whether the data is loaded from disk and needs to be saved or has been saved already
    SAVE_DATA = True
    if SAVE_DATA:
        os.makedirs('../data') # make a directory that would contain the saved files

    if SAVE_DATA:
        X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMG_DIR)
        np.save("../data/X_train.npy", X_train)
        np.save("../data/y_train.npy", y_train)
        np.save("../data/X_val.npy", X_val)
        np.save("../data/y_val.npy", y_val)
        np.save("../data/X_test.npy", X_test)
        np.save("../data/y_test.npy", y_test)
    else:
        X_train = np.load("D:/data/X_train.npy") # load saved data depending where the saved data is from. 
        y_train = np.load("D:/data/y_train.npy")
        X_val = np.load("D:/data/X_val.npy")
        y_val = np.load("D:/data/y_val.npy")
        X_test = np.load("D:/data/X_test.npy")
        y_test = np.load("D:/data/y_test.npy")
      

    # create the model
    model = Perceiver(data_size=73 * 64,  # size of data in rows multiplied by column
                      latent_size=256,  # size of latent array specified in paper
                      proj_size=27,  # size of projection after encoding, also equivalent to 2*(2*num_bands + 1) + 1
                      num_heads=8,  # The number of transformer head
                      num_trans_blocks=6,  # Number of transformer blocks used in the paper
                      num_iterations=8,  # Repetition of cross-attention and transformer modules used in the paper
                      max_freq=10,  # Maximum frequency for fourier encoding
                      num_bands=6,  # Number of bands for fourier encode used in paper
                      lr=0.001,  # The learning rate for the optimiser
                      epoch=10,  # Number of epochs for training process
                      weight_decay=0.0001,  # Decay weight for the optimiser
                      )

    # Saving the model if required, but can be omitted. Uncomment if model needs to be saved
    # save_dir = './model_save'
    # save_model = tf.train.Checkpoint(
    #         model=model)

    # model_save_manager = tf.train.CheckpointManager(save_model, save_dir, max_to_keep=3)

    # Now training the model with the given the training, validation, test and batch size
    # as well as saving the model
    training_history = model.train(
                    (X_train, y_train),
                    (X_val, y_val),
                    (X_test, y_test),
                    32)

    # model_save_manager.save()

    # Plotting the learning curves
    plot_learning_curve(training_history)
    plot_prediction(model, (X_test, y_test))

    
