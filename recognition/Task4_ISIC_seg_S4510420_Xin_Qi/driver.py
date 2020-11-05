__author__ = "Xin Qi"

import tensorflow as tf
import zipfile
import glob
import random
import numpy as np

from model import *

def main():
    """The driver script of using the improved Unet model"""
    # download the dataset
    dataset_url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=f0d763f9-d847-4150-847c-e0ec92d38cc5&files_ids=10200257"
    data_path = tf.keras.utils.get_file(origin=dataset_url,
                                    fname = "ISIC2018_Task1-2_Training_Data.zip")

    with zipfile.ZipFile(data_path) as zf:
        zf.extractall()

    image = sorted(glob.glob('ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
    mask = sorted(glob.glob('ISIC2018_Task1_Training_GroundTruth_x2/*.png'))

    # N_sample = len(mask)
    N_train = 1800
    N_vali = 400
    N_test = 394

    train_image = []
    train_mask = []
    vali_image = []
    vali_mask = []
    test_image = []
    test_mask = []

    random.seed(42)
    random.shuffle(image)
    random.seed(42)
    random.shuffle(mask)
    # split dataset into training set, validation set and testing set
    train_image = image[:N_train]
    train_mask = mask[:N_train]
    vali_image = image[N_train:N_train+N_vali]
    vali_mask = mask[N_train:N_train+N_vali]
    test_image = image[-N_test:]
    test_mask = mask[-N_test:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_image,train_mask))
    val_ds = tf.data.Dataset.from_tensor_slices((vali_image,vali_mask))
    test_ds = tf.data.Dataset.from_tensor_slices((test_image,test_mask))

    def map_fn(filename, label):
        # Load the raw data from the file as a string.
        img = tf.io.read_file(filename)
        # Convert the compressed string to a 3D uint8 tensor.
        img = tf.image.decode_png(img, channels=3)
        # resize the image size to the same size
        img = tf.image.resize(img, (256, 256))
        # Standardise values to be in the [0, 1] range.
        img = tf.cast(img, tf.float32) / 255.0

        # Binarilize the mask
        label = tf.io.read_file(label)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, (256, 256))
        label =tf.cast(label>0, tf.int8)
        
        return img, label

    # Use Dataset.map to apply this transformation.
    train_ds = train_ds.map(map_fn)
    val_ds = val_ds.map(map_fn)
    test_ds = test_ds.map(map_fn)
    # buile the model
    new_Unet = improved_UNET((256,256,3), train_ds, val_ds)
    new_Unet.compile()
    new_Unet.fit(batch_size=8, epoch=15, checkpoint=True, checkpoint_name='ISIC.hdf5')

    # print the DSC on testing set.
    image, label = next(iter(test_ds.batch(394)))
    predict_result = new_Unet.predict(image)

    dsc_result = []
    for i in range(394):
        dsc_result.append(new_Unet.dice_coef(label[i], predict_result[i]))

    print("The DSC of the improved UNET on testing set is: ", np.mean(dsc_result))


    # plot the raw_image, ground truth and predict result together to check the actual performance
    # pick one image randomly from training set, validation set and testing set respectively.
    train_image, train_mask = next(iter(train_ds.batch(50)))
    new_Unet.show_result(train_image[30], train_mask[30], 'train_set_result.jpg')

    val_image, val_mask = next(iter(val_ds.batch(50)))
    new_Unet.show_result(val_image[10], val_mask[10], 'val_set_result.jpg')

    test_image, test_mask = next(iter(test_ds.batch(50)))
    new_Unet.show_result(test_image[20], test_mask[20], 'test_set_result.jpg')

if __name__ == "__main__":
    main()

