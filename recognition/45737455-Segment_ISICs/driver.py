import glob
import random
import numpy as np
import tensorflow as tf
from model import *
import matplotlib.pyplot as plt


def sanity_check(input_filenames, groundtruth_filenames):
    """
    Sanity check of the datasets.
    1. Checking if the number of input files is the same as the number of ground truth files.
    2. Checking if the serial number of input file is the same as that of ground truth files.

    Parameters
    ----------
    input_filenames : list
        A list of input image file names.
    groundtruth_filenames : list
        A list of ground truth image file names.
    """
    assert len(input_filenames) == len(groundtruth_filenames), \
        "the number of input files does not equal to the number of groundtruth files"

    random_index = random.randint(0, len(input_filenames) - 1)
    random_input, random_gtruth = input_filenames[random_index], groundtruth_filenames[random_index]

    input_number_index, gtruth_number_index = random_input.index("ISIC_") + len("ISIC_"), \
                                              random_gtruth.index("ISIC_") + len("ISIC_")

    assert random_input[input_number_index:input_number_index + 7] == \
           random_gtruth[gtruth_number_index:gtruth_number_index + 7], \
        "the image id of input file and groundtruth file does not match"


def train_val_test_split(inputs, groundtruths, split_rate=.1):
    """
    Splits the dataset into train/val/test set, the ratio is 8:1:1 by default.
    Converts them into Dataset format of tensorflow.

    Parameters
    ----------
    inputs : list or ndarray
        A list of input image file names.
    groundtruths : list or ndarray
        A list of ground truth image file names.
    split_rate : float, default=0.1
        The proportion of the validation set and test set.

    Returns
    -------
    train_ds : tf.data.Dataset
        The training set.
    val_ds : tf.data.Dataset
        The validation set.
    test_ds : tf.data.Dataset
        The test set.
    """
    if inputs.__class__.__name__ == 'list':
        inputs = np.array(inputs)
    if groundtruths.__class__.__name__ == 'list':
        groundtruths = np.array(groundtruths)
    test_size = int(len(inputs) * split_rate)

    indices = np.random.permutation(len(inputs))
    train_idx, val_idx, test_idx = indices[2 * test_size:], \
                                   indices[test_size:2 * test_size], \
                                   indices[:test_size]

    train_images, train_labels = inputs[train_idx], groundtruths[train_idx]
    val_images, val_labels = inputs[val_idx], groundtruths[val_idx]
    test_images, test_labels = inputs[test_idx], groundtruths[test_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    return train_ds, val_ds, test_ds


def map_fn(image_filename, label_filename):
    """
    The map function that load the input image files and ground truth image files
    and pre-process the data.
    """
    # input image
    img = tf.io.read_file(image_filename)
    img = tf.io.decode_jpeg(img, channels=3)  # RGB image
    img = tf.image.resize(img, (256, 256))
    img = img / 255.

    # ground truth image
    label = tf.io.read_file(label_filename)
    label = tf.io.decode_jpeg(label, channels=1)  # greyscale image
    label = tf.image.resize(label, (256, 256))
    label = label / 255.
    label = tf.cast(label > 0.5, dtype=tf.float32)

    return img, label


def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Calculates the dice similarity coefficient for comparing the similarity of two batch of data.
    See more in https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values of y.
    y_pred : tf.Tensor
        The predicted values of y.
    smooth : float, default=1e-5
        A small value that will be added to the numerator and denominator.

    Returns
    -------
    dice : tf.Tensor
        The dice similarity coefficient value.
    """
    intersection = tf.reduce_sum(y_pred * y_true)
    X = tf.reduce_sum(y_pred)
    Y = tf.reduce_sum(y_true)

    dice = (2. * intersection + smooth) / (X + Y + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    """
    Creates the loss function with respect to the dice coefficient.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values of y.
    y_pred : tf.Tensor
        The predicted values of y.

    Returns
    -------
    loss : tf.Tensor
        The loss with respect to the dice coefficient value.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def test_result(model, test_ds):
    first = True
    for test_input, test_true in test_ds:
        test_predict = model.predict(test_input)
        test_predict = np.where(test_predict < 0.5, 0, 1)

        test_predict = test_predict[:, :, :, 0]
        test_true = test_true.numpy()[:, :, :, 0]
        test_input = test_input.numpy()

        if first:
            first = False
            test_input_lst = test_input
            test_true_lst = test_true
            test_predict_lst = test_predict
        else:
            test_input_lst = np.append(test_input_lst, test_input, axis=0)
            test_true_lst = np.append(test_true_lst, test_true, axis=0)
            test_predict_lst = np.append(test_predict_lst, test_predict, axis=0)

    assert test_true_lst.shape == test_predict_lst.shape, \
        "the shape of the predicted result is different from the shape of groundtruth image"

    return test_input_lst, test_true_lst, test_predict_lst


def preprocessed_visualization(input_img, gtruth_img):
    # visualize input images and groundtruth images
    fig = plt.figure(figsize=(10, 8))
    rows, columns = 2, 3

    for i in range(3):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(input_img[i])
        plt.title("input sample " + str(i+1))
        plt.axis('off')

        fig.add_subplot(rows, columns, i+1+columns)
        plt.imshow(gtruth_img[i], cmap='gray')
        plt.title("ground truth sample " + str(i+1))
        plt.axis('off')

    plt.show()


def test_visualization(input_lst, true_lst, predict_lst):
    # visualize test result
    fig = plt.figure(figsize=(10, 10))
    rows, columns = 3, 3

    for i in range(3):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(input_lst[i])
        plt.title("input image "+str(i+1))
        plt.axis('off')

        fig.add_subplot(rows, columns, i+1+columns)
        plt.imshow(true_lst[i], cmap='gray')
        plt.title("ground truth "+str(i+1))
        plt.axis('off')

        fig.add_subplot(rows, columns, i+1+2*columns)
        plt.imshow(predict_lst[i], cmap='gray')
        plt.title("predicted result "+str(i+1))
        plt.axis('off')

    plt.show()


def main():
    # data file directories
    dataset_dir = 'ISIC2018_Task1-2_Training_Data/'
    input_dir = dataset_dir + 'ISIC2018_Task1-2_Training_Input_x2/'  # directory of original images
    groundtruth_dir = dataset_dir + 'ISIC2018_Task1_Training_GroundTruth_x2/'  # directory of groundtruth images

    # load data files
    print("\n============= Data Files Loading =============")
    input_filenames = glob.glob(input_dir + '/*.jpg')
    input_filenames = sorted(input_filenames)

    groundtruth_filenames = glob.glob(groundtruth_dir + '/*.png')
    groundtruth_filenames = sorted(groundtruth_filenames)

    sanity_check(input_filenames, groundtruth_filenames)
    print("[done] load data files and sanity check.")

    # data preprocessing
    print("\n============= Data Preprocessing =============")
    train_ds, val_ds, test_ds = train_val_test_split(input_filenames, groundtruth_filenames)

    train_ds = train_ds.map(map_fn)
    val_ds = val_ds.map(map_fn)
    test_ds = test_ds.map(map_fn)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)
    test_ds = test_ds.batch(32)
    print("[done] read images and split into training/validation/test sets.")

    # visualization of images
    # for input_img, gtruth_img in val_ds.take(1):
    #     preprocessed_visualization(input_img.numpy(), gtruth_img.numpy())
    #     break

    # train improved U-Net model
    print("\n=============== Model Training ===============")
    model = improved_UNet()
    # model.summary()
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])
    model.fit(x=train_ds, epochs=5, validation_data=val_ds, shuffle=True)
    print("[done] train the improved U-Net model.")

    # check the performance of the trained model
    print("\n================ Model Testing ================")
    test_input_lst, test_true_lst, test_predict_lst = test_result(model, test_ds)
    print("Test data contains {} images.".format(test_input_lst.shape[0]))

    test_predict = tf.convert_to_tensor(test_predict_lst, dtype=tf.float32)
    test_true = tf.convert_to_tensor(test_true_lst, dtype=tf.float32)
    print("The testing dice similarity coefficient is:", round(dice_coef(test_true, test_predict).numpy(), 3))

    # visualization of testing result
    test_visualization(test_input_lst, test_true_lst, test_predict_lst)


if __name__ == "__main__":
    main()
