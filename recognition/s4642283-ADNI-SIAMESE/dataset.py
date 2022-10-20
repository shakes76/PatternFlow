import numpy as np 
import os
from PIL import Image
import random

from sklearn.model_selection import train_test_split

# This function borrows logic from https://keras.io/examples/vision/siamese_contrastive/
def make_pairs(x, y):
    """
    Creates a tuple containing pairs of images, where images belonging to
    the same class are labelled 0 while images that belong to different classes
    are labelled 1.
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

def data_loader():
    """
    Function that loads in data from the directory, splits the data into
    training, validation and testing and makes pairs of data and labels them
    to feed the Siamese network.
    """
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    X_training = []
    X_train_labels = []
    for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/AD")):
        fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/AD", fname)
        im = Image.open(fpath)
        X_training.append(np.array(im))
        X_train_labels.append(1)
        im.close()

    for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/NC")):
        fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/NC", fname)
        im = Image.open(fpath)
        X_training.append(np.array(im))
        X_train_labels.append(0)
        im.close()

    # Convert to numpy array
    X_training = np.array(X_training)
    X_train_labels = np.array(X_train_labels)
    print(X_training.shape)

    x_test = []
    y_test = []
    for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/AD")):
        fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/AD", fname)
        im = Image.open(fpath)
        x_test.append(np.array(im))
        y_test.append(1)
        im.close()

    for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/NC")):
        fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/NC", fname)
        im = Image.open(fpath)
        x_test.append(np.array(im))
        y_test.append(0)
        im.close()

    # Convert to nparray
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Make validation set
    x_train, x_val, y_train, y_val = train_test_split(X_training, X_train_labels, test_size=0.2, random_state=8, shuffle=True)

    # Make train pairs
    pairs_train, labels_train = make_pairs(x_train, y_train)

    # Make validation pairs
    pairs_val, labels_val = make_pairs(x_val, y_val)

    # Make test pairs
    pairs_test, labels_test = make_pairs(x_test, y_test)

    # Create pairs to feed Left and Right Siamese Networks
    x_train_1 = pairs_train[:, 0]
    x_train_2 = pairs_train[:, 1]

    x_val_1 = pairs_val[:, 0]
    x_val_2 = pairs_val[:, 1]

    x_test_1 = pairs_test[:, 0]
    x_test_2 = pairs_test[:, 1]

    return (x_train_1, x_train_2), labels_train, (x_val_1, x_val_2), labels_val, (x_test_1, x_test_2), labels_test
