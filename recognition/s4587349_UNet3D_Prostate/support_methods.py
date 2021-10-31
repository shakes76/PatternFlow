import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from tensorflow import keras
import driver as drv

# print(tf.__version__)

# todo need to input X_set, y_set whish are x_
class ProstateSequence(keras.utils.Sequence):
    # inspiration
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    # https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
    """Data generator"""


    def __init__(self, x_set, y_set, batch_size=1, training = True):
        # def __init__(self, x_set, y_set, batch_size=1, dim=(256, 256, 128), n_channels=1,
        #              n_classes=6, shuffle=False):
        """
        :param x_set: list of images
        :param y_set: list of labels
        :param batch_size: images per batch, default set to 1
        :param dim: dimensions of image
        :param n_channels: number of channels
        :param n_classes: number of classes, 6 for this data set
        :param shuffle: Whether to shuffle the data to limit learning of order
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.dim = (256, 256, 128)
        self.n_channels = 1
        self.n_classes = 6
        self.shuffle = False   # True may help reduce overfitting
        self.on_epoch_end()
        self.training = training


    def __len__(self):
        """Number of batches per epoch"""
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):  # todo setup for shuffle
        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
        """
        Gets one batch
        :param idx:
        :return: 1 batch of data and a matching batch of labels
        """
        # create tmp list of image/label names for batch
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_data_tmp = [self.x[k] for k in indexes]
        list_label_tmp = [self.y[k] for k in indexes]
        # generate data
        X = self._generation_x(list_data_tmp)
        y = self._generation_y(list_label_tmp)
        X = X[:, :, :, :, np.newaxis]

        X = X.astype("float32")         # 0 upwards, max for these images ~1300
        y = y.astype("uint8")           # num 0-5

        # Normalise X: (X-mean)/stdev
        X_norm = z_norm(X)

        if self.training:
            return X_norm, y     # for training and validation generation
        else:
            return X_norm        # If running test gen, then only need X


    def _generation_x(self, list_data_tmp):
        """
        Generates one batch of data, given a list of data file paths/names.
        :param list_data_tmp:
        :return: One batch of nparray of data.
        """
        X = np.empty((self.batch_size, *self.dim))

        for i, id in enumerate(list_data_tmp):
            X[i,] = self.read_nii(id)
        return X

    def _generation_y(self, list_label_tmp):
        """
        Generates one batch of labels, given a list of labels file paths/names.
        The labels match the data files from _generation_x()
        :param list_data_tmp:
        :return: a single batch in an nparray
        """
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, id in enumerate(list_label_tmp):
            y2 = self.read_nii(id)
            y[i,] = tf.keras.utils.to_categorical(y2, num_classes=self.n_classes,
                                                  dtype='uint8')
        return y

    def read_nii(self, file_path):
        """ Reads and returns nparray data from single .nii image"""
        img = nib.load(file_path)
        img_data = img.get_fdata()
        return img_data

    def on_epoch_end(self):
        'Shuffles indexes at end of each epoch and initalise self . indexes'
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def plot_loss(history):
    """
    Plots training and validation losses

    :param history: from model.fit()
    :return: outputs a figure of the plot
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    plt.close()


def plot_accuracy(history):
    """
    Plot training and validation accuracy

    :param history: from model.fit()
    :return: outputs a figure of the plot
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    plt.close()


def data_info():
    """ Prints info on original data and labels files via raw_data_info(image).  """
    # data info
    img_mr = (nib.load(drv.X_TRAIN_DIR + '\\Case_004_Week0_LFOV.nii.gz')).get_fdata()
    raw_data_info(img_mr)
    # label info
    img_label = (nib.load(drv.Y_TRAIN_DIR + '\\Case_004_Week0_SEMANTIC_LFOV.nii.gz')).get_fdata()
    raw_data_info(img_label)


def raw_data_info(image):
    """ prints info of provided image. """
    print("image information")
    print(type(image))
    print(image.dtype)
    print(image.shape)
    print(np.amin(image), np.amax(image))
    print()


def slices(img):
    """ takes slices of input image and sends them to be printed to screen."""
    slice_0 = img[127, :, :]
    slice_1 = img[:, 127, :]
    slice_2 = img[:, :, 63]
    show_slices([slice_0, slice_1, slice_2])

def slices_ohe(img):
    """ takes slices of one hot encoded label and sends them to be printed to
    screen. Looking for slices that include a portion of each class """
    slice_0 = img[:,128,:,0]  # Background
    slice_1 = img[:,128,:,1]  # Body
    slice_2 = img[:,128,:,2]  # Bones
    slice_3 = img[:,128,:,3]  # Bladder
    slice_4 = img[:,150,:,4]  # Rectum
    slice_5 = img[:,128,:,5]  # Prostate

    show_slices([slice_0, slice_1, slice_2, slice_3, slice_4, slice_5])


def slices_pred(img, filename, title):
    """
    Save (and display) a slice of either y_predict or y_true.
    :param title: Title for image
    :param img: Image to print
    :param filename: Name to save file
    :return: Nothing
    """
    slice_0 = img[0,:,127,:,2]
    plt.imshow(slice_0)
    plt.title(title)
    show_slices([slice_0])
    plt.savefig(filename)
    plt.close()

def show_slices(sliced):
    """
    Print to screen from a given list of slices.

    :param sliced: a list of slices of image
    :return: Nothing
    """
    for i in sliced:
        plt.imshow(i.T)
        plt.show()
        # plt.close()

def dice_coef(y_true, y_pred):
    """
    Calculates DSC

    :param y_true: array of all images in label_test
    :param y_pred: Array output from pred_argmax
    :return: return
    """
    smooth = 0.1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multiclass(y_true, y_pred, classes):
    """
    Inspired by gattia, 7 April 2018
    https://github.com/keras-team/keras/issues/9395

    :param y_true: labels for test data
    :param y_pred: predicted labels for test data
    :param classes: 6 classes in data set
    :return: a list containing DSC values for each class
    """
    dice = []
    for index in range(classes):
        x = dice_coef(y_true[:,:,:, :,index], y_pred[:,:,:,:,index])
        dice = dice + [x]
    print_dice(dice)
    return dice


def print_dice(dice):
    """
    Prints the dice coefficient for each of the 6 classes.

    :param dice:
    :return:
    """
    print("Dice Coef Background: ", dice[0])
    print("Dice Coef Body: ", dice[1])
    print("Dice Coef Bones: ", dice[2])
    print("Dice Coef Bladder: ", dice[3])
    print("Dice Coef Rectum: ", dice[4])
    print("Dice Coef Prostate: ", dice[5])
    print("DSC list: ", dice)
    print("Average DSC: ", sum(dice) / len(dice))


def model_summary_print(summ):
    """ Prints model.summary()  to file.
    :param: print_fn call
    """
    with open('model_summary.txt', 'w+') as f:
        print(summ, file=f)


def dim_per_directory():
    """ iterates through data and label directories, checking that dimensions
    are as expected."""
    print("image_train")
    dim_check(drv.image_train)
    print("image_validate")
    dim_check(drv.image_validate)
    print("image_test")
    dim_check(drv.image_test)
    print("label_train")
    dim_check(drv.label_train)
    print("label_validate")
    dim_check(drv.label_validate)
    print("label_test")
    dim_check(drv.label_test)


def dim_check(filepath):
    """ Expected dim of each image and label is (256,256,128)
    Case_019_week1 has dimensions (256,256,144)
    :param filepath:
    :return: None
    """
    for i in filepath:
        tups = read_nii(i).shape
        x, y, z = tups
        if x != 256:
            print("flag x: ", i, x)
        if y != 256:
            print("flag y: ", i, y)
        if z != 128:
            print("flag z: ", i, z)


def min_max_value(file_path):
    """ Return min and max voxel value of all images in file path"""
    min_value = 1000
    max_value = 0
    for i in file_path:
        maxv = np.amax(read_nii(i))
        minv = np.amin(read_nii(i))
        if maxv > max_value:
            max_value = maxv
        if minv < min_value:
            min_value = minv
    return min_value, max_value


def read_nii(filepath):
    """ Reads and returns nparray data from single .nii image"""
    img = nib.load(filepath)
    img_data = img.get_fdata()
    return img_data


def normalise(image):  
    """ If minv = 0, then is equiv to dividing all values by image maximum value
    :param image: data image
    :param minv: minimum voxel value
    :param maxv: maximum voxel value
    :return: normalised image
    """
    maxv = np.amax(image)
    minv = np.amin(image)
    img_norm = (image - minv) / (maxv - minv)
    img_norm = img_norm.astype("float64")
    return img_norm


def normalise2(path):
    """
    Normalises image given path to nii.gz file.

    :param path: Path to image
    :return:  normalised image
    """
    image = read_nii(path)
    maxv = np.amax(image)
    minv = np.amin(image)
    img_norm = (image - minv) / (maxv - minv)
    img_norm = img_norm.astype("float64")
    return img_norm


def z_norm(image):
    """ Returns z normalised image. This will involve negative values. May
    require adjusted colour palette to display.
    :param image:
    :return: z normalised image
    """
    return (image - np.mean(image)) / np.std(image)


