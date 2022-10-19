import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from matplotlib import pyplot as plt

def process_data():
    """
    imports the preprocessed ADNI MRI dataset from https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI and returns 
    the tuple including (train, validation, test) data set
    """

    # load the train and test data from the dataset directory 
    # note that the data has already been preprocessed into train and test
    train_ds= image_dataset_from_directory(
        r"/home/Student/s4642548/AD_NC/train",
        image_size=(256, 256),
        label_mode=None,
        color_mode = "grayscale"
    )
    
    # use the test dataset for validation
    valid_ds = image_dataset_from_directory(
        r"/home/Student/s4642548/AD_NC/test",
        image_size=(256, 256),
        label_mode=None,
        color_mode = "grayscale"
    )
    
    # use a batch of 32 from validation to test predications after training is done in test dataset
    test_ds = valid_ds.take(1)
    valid_ds = valid_ds.skip(1)

    # Scale to (0, 1)
    train_ds = train_ds.map(scale)
    valid_ds = valid_ds.map(scale)
    test_ds = test_ds.map(scale)


    # map all datasets to include 4x downsample version to be used as input into the model as well as the original image                           
    train_ds = train_ds.map(
        lambda x: (resize_input(x), x)
    )
    
    train_ds = train_ds.prefetch(buffer_size=32)
    
    valid_ds = valid_ds.map(
        lambda x: (resize_input(x), x)
    )
    
    valid_ds = valid_ds.prefetch(buffer_size=32)
    test_ds = test_ds.map(
            lambda x: (resize_input(x), x)
    )
    
    test_ds = test_ds.prefetch(buffer_size=32)
    
    return train_ds, valid_ds, test_ds

def scale(image):
    """
    scales the values in an image array from [0,255] to [0,1]
    """
    image = image / 255.0
    return image 

def resize_input(input):
    """
    down samples a 256 by 256 image by a factor of 4 to a 64 by 64 image
    """
    return tf.image.resize(input, [256//4, 256//4], method="area")

