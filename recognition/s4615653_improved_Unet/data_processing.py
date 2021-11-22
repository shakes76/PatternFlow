import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt


def load_data():
    # Getting the file paths of data and sort paths
    inputs = sorted(glob.glob("D:/comp3710/s4615653_improved_Unet/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg"))
    truth = sorted(glob.glob("D:/comp3710/s4615653_improved_Unet/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png"))

    return inputs, truth


def decode_jpeg(inputs_fn):
    # function used to decode jpeg
    jpeg = tf.io.read_file(inputs_fn)
    jpeg = tf.image.decode_jpeg(jpeg, channels=3)
    jpeg = tf.image.resize(jpeg, (256, 256))
    return jpeg

def decode_png(truth_fn):
    # function used to decode png
    png = tf.io.read_file(truth_fn)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png

def map_fn(inputs_fp,truth_fp):
    # map the data
    inputs = decode_jpeg(inputs_fp)
    inputs = tf.cast(inputs, tf.float32)/255.0

    truth = decode_png(truth_fp)
    truth = truth == [0,255]
    truth = tf.cast(truth, tf.uint8)

    return inputs, truth

def split_data(inputs,truth,ratio_train,ratio_validation,ratio_test):
    # split the data into training,test and validation dataset.
    num_inputs = len(inputs)

    # The number of images in our validation and test set
    val_size = int(num_inputs * ratio_validation)
    test_size = int(num_inputs * ratio_test)
    train_size = int(num_inputs * ratio_train)

    # array of inputs for the training images
    train_inputs = inputs[:train_size]
    # array of inputs for validation images
    val_inputs = inputs[train_size:train_size + val_size]
    # array of inputs for test images
    test_inputs = inputs[-test_size:]

    # array of truth for the training images
    train_truth = truth[:train_size]
    # array of truth for validation images
    val_truth = truth[train_size:train_size + val_size]
    # array of truth for test images
    test_truth = truth[-test_size:]

    return train_inputs, train_truth, val_inputs, val_truth, test_inputs, test_truth

# create TensorFlow Datasets and shuffle them
def tensor_data(train_inputs, train_truth, val_inputs, val_truth, test_inputs, test_truth):
    # make the tensor data
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_truth))
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_truth))
    test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_truth))

    return  train_ds, val_ds, test_ds

def shuffle_data(train_ds,val_ds,test_ds,train_inputs,val_inputs,test_inputs):
    #shuffle the data
    train_ds = train_ds.shuffle(len(train_inputs))
    val_ds = val_ds.shuffle(len(val_inputs))
    test_ds = test_ds.shuffle(len(test_inputs))

    return train_ds, val_ds, test_ds


def map_data(train_ds,val_ds,test_ds):
    # map data to data arrays
    train_ds = train_ds.map(map_fn)
    val_ds = val_ds.map(map_fn)
    test_ds = test_ds.map(map_fn)

    return train_ds, val_ds, test_ds

def data_processing():
    # a function will be called in main.py to process data processing.

    inputs, truth = load_data()

    train_inputs, train_truth, val_inputs, val_truth, test_inputs, test_truth = split_data(inputs, truth, 0.7, 0.1, 0.2)

    train_ds, val_ds, test_ds = tensor_data(train_inputs, train_truth, val_inputs, val_truth, test_inputs, test_truth)

    train_ds, val_ds, test_ds = shuffle_data(train_ds, val_ds, test_ds, train_inputs, val_inputs, test_inputs)

    train_ds, val_ds, test_ds = map_data(train_ds,val_ds,test_ds)

    return train_ds, val_ds, test_ds



def main():
    # load data and test if data loaded correctly?
    inputs,truth = load_data()
    print(len(inputs))

    train_inputs, train_truth, val_inputs, val_truth, test_inputs, test_truth = split_data(inputs,truth,0.7,0.1,0.2)

    train_ds, val_ds, test_ds = tensor_data(train_inputs,train_truth,val_inputs,val_truth,test_inputs,test_truth)

    train_ds, val_ds, test_ds = shuffle_data(train_ds, val_ds, test_ds, train_inputs, val_inputs, test_inputs)

    train_ds = train_ds.map(map_fn)

    val_ds = val_ds.map(map_fn)

    test_ds = test_ds.map(map_fn)

#View loaded data to see if data is loaded correctly
    for input, truth in train_ds.take(1):
        print(input)
        print(truth)
        plt.figure(figsize=(10, 10))
        View_list = [tf.squeeze(input), tf.argmax(truth, axis=-1)]
        print(View_list)
        print(len(View_list))
        for i in range(len(View_list)):
            plt.subplot(1, len(View_list), i + 1)
            plt.imshow(View_list[i], cmap='gray')
            if i == 0:
                plt.title("Input Image")
            if i == 1:
                plt.title("truth Image")
            plt.axis('off')
        plt.show()

#Run the main function.
if __name__ == "__main__":
    main()



