"""
Driver file for UNet3D training and evaluating with
Pelvis MRI dataset (provided in folder ./dataset/)

@author Cody Baldry
@student_number 45804290
@date 1 November 2021
"""
# imports
import tensorflow as tf
from tensorflow import keras as K

from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from math import floor, ceil
import glob
import re

from model import unet_3d

# scans data and labels folders for files
def getScans(dataFolder, labelsFolder):
    dataFiles = sorted(glob.glob(dataFolder + "*"))
    labelsFiles = sorted(glob.glob(labelsFolder + "*"))
    
    return list(zip(dataFiles, labelsFiles))

# lazily convert paths to files into images / labels
def im_file_to_tensor(dataPath, labelPath, fileNames, channels, n_classes, shape):
    def _im_file_to_tensor(dataPath, labelPath):
        dp, lp = fileNames[dataPath]
        image = tf.convert_to_tensor(nib.load(dp).get_fdata().astype('float32'))
        image = tf.cast(image, tf.float32)
        image = tf.stack((image,)*channels, axis=-1)
        image = tf.expand_dims(image, axis=0)
        labels = tf.convert_to_tensor(nib.load(lp).get_fdata().astype('float32'))
        labels = to_categorical(tf.expand_dims(labels, axis=3), num_classes=n_classes)
        labels = tf.cast(labels, tf.float32)
        labels = tf.expand_dims(labels, axis=0)
        return image, labels
    data, label = tf.py_function(_im_file_to_tensor, 
                                 inp=(dataPath, labelPath), 
                                 Tout=(tf.float32, tf.float32))
    data.set_shape((1,) + shape + (channels,))
    label.set_shape((1,) + shape + (n_classes,))
    return data, label

# dice coefficient function
def dice_coefficient(y_true, y_pred):
    y_true = K.backend.flatten(y_true)
    y_pred = K.backend.flatten(y_pred)
    intersection = K.backend.sum(K.backend.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + 1) / (K.backend.sum(K.backend.square(y_true),-1)
            + K.backend.sum(K.backend.square(y_pred),-1) + 1)

# dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def main():
    # model parameters
    shape = nib.load("./dataset/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz").get_fdata().shape
    channels = 3
    n_classes = 6

    LR = 0.0001
    batch_size = 1
    batch_num = 15
    optim = K.optimizers.Adam(LR)

    fileNames = getScans("./dataset/semantic_MRs_anon/",
        "./dataset/semantic_labels_anon/")
    total_images = len(fileNames)

    # get each group of cases' positions
    ind_of_first_case = []
    i = 0
    lastCase = None
    for path, _ in fileNames:
        x = re.search("Case_(\d+)_", path).group(1)
        if lastCase != x:
            ind_of_first_case.append(i)
            lastCase = x
        i += 1

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(range(len(fileNames))), tf.constant(range(len(fileNames)))))

    dataset = dataset.map((lambda x, y : im_file_to_tensor(x, y, fileNames, channels, n_classes, shape)),
            num_parallel_calls=tf.data.AUTOTUNE)

    # train/test/val split
    i_train = floor(0.7 * total_images)
    while i_train not in ind_of_first_case:
        i_train += 1
    rem = total_images - i_train - 1
    i_test = floor(i_train + rem/2)
    while i_test not in ind_of_first_case:
        i_test += 1
    i_test -= i_train

    train_ds = dataset.take(i_train)
    test_ds = dataset.skip(i_train).take(i_test)
    val_ds = dataset.skip(i_train + i_test)

    print("train", train_ds.cardinality())
    print("test", test_ds.cardinality())
    print("validation", val_ds.cardinality())

    print("Successfully created, processed and split dataset!")

    # using metrics IOU and dice score to evaluate
    metrics = [K.metrics.MeanIoU(num_classes=n_classes), dice_coefficient]

    # create model
    new_shape = shape + (channels,)
    model = unet_3d(new_shape, n_classes)

    # compile model
    model.compile(optimizer = optim, loss=dice_loss, metrics=metrics)
    print(model.summary())

    # fit model with train_ds, validate on val_ds
    history = model.fit(train_ds, batch_size=batch_size, epochs=batch_num, verbose=1,
            validation_data=val_ds)

    model.save(f'./segment_out/3D_unet_{batch_num}epochs.h5')

    # plot the performance of the model over time
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./segment_out/loss.png")
    plt.close()

    acc = history.history['dice_coefficient']
    val_acc = history.history['val_dice_coefficient']

    plt.plot(epochs, acc, 'b', label='Training Dice')
    plt.plot(epochs, val_acc, 'g', label='Validation Dice')
    plt.title('Training and validation Dice Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.savefig("./segment_out/dice.png")
    plt.close()

    # evaluate performance against test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(test_loss, test_acc)

    # save a test comparison slice for visualisation
    import random
    test_img_number = random.randint(0, len(test_ds))
    test_img = None
    ground_truth = None
    for image, label in test_ds.take(1):
        test_img = image
        ground_truth = label

    test_pred1 = model.predict(test_img)
    test_prediction1 = tf.argmax(test_pred1, axis=4)[0,:,:,:]
    ground_truth_argmax = tf.argmax(ground_truth, axis=4)

    test_slice = random.randint(floor(128 * (1/3)), ceil(128 * (2/3)))

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[0, :,:,test_slice,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth_argmax[0,:,:,test_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction1[:,:,test_slice])
    plt.savefig("./segment_out/compare.png")

if __name__ == "__main__":
    main()