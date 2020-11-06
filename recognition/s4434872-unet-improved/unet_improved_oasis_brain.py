"""
OASIS Brain Dataset Segmentation with Improved UNet, 
with all labels having a minimum Dice Similarity Coefficient 
of 0.9 on the test set.

@author Dhilan Singh (44348724)

Start Date: 01/11/2020
"""
import tensorflow as tf
import glob
import preprocess

print('Tensorflow Version:', tf.__version__)

# Download the dataset (use the direct link given on the page)
print("> Loading images ...")
# tf.keras.utils.get file downloads a file from a URL if it not already in the cache.
#     origin: Original URL of the file.
#     fname: Name of the file. If an absolute path /path/to/file.txt is specified the 
#            file will be saved at that location (in cache directory).
#            NEEDS FILE EXTENSION TO WORK!!!
#     extract: If true, extracting the file as an Archive, like tar or zip.
#     archive_format: zip, tar, etc...
#     returns: Path to the downloaded file.
dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
data_path = tf.keras.utils.get_file(origin=dataset_url,
                                    fname="keras_png_slices_data.zip",
                                    extract=True,
                                    archive_format="zip")

# Remove the .zip file extension from the data path
data_path_clean = data_path.split('.zip')[0]

# Load filenames into a list in sorted order
train_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob(data_path_clean +"/keras_png_slices_seg_train/*.png"))
val_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_validate/*.png"))
test_images = sorted(glob.glob(data_path_clean + "/keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob(data_path_clean + "/keras_png_slices_seg_test/*.png"))

# Build tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

# Make the dataset to be reshuffled each time it is iterated over.
# This is so that we get different batches for each epoch.
# For perfect shuffling, the buffer size needs to be greater than or equal to 
# the size of the dataset.
train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))

# Use Dataset.map to apply preprocessing transformation.
# Normalize the images and pixel-wise one-hot encode the segmentation masks.
print("> Preprocessing images ...")
train_ds = train_ds.map(preprocess.process_path)
val_ds = val_ds.map(preprocess.process_path)
test_ds = test_ds.map(preprocess.process_path)

# Input Image Parameters for model
image_pixel_rows = 256 
image_pixel_cols = 256
image_channels = 1

from tensorflow.keras.layers import Input, Activation, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape, Dropout, InstanceNormalization
from tensorflow.keras.initializers import GlorotNormal

def conv2D_layer(input_layer, 
                 n_filters, 
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=ReLU(), 
                 use_bias=True,
                 kernel_initializer=GlorotNormal(),
                 batch_normalization=False,
                 instance_normalization=False
                 **kwargs):
    """
    Create a 2D convolutional layer according to parameters.

    @param input_layer:
        The input layer.
    @param n_filters:
        The number of filters.
    @param kernel_size:
        The size of the kernel filter.
    @param strides:
        The stride number during convolution.
    @param activation:
        Keras activation layer to use.
    @param batch_normalization:
        If true, apply batch normalization.
    @param instance_normalization:
        If truem apply instance normalization.

    Reference: Adapted from Shakes lecture code layers.py
    """
    # Create a 2D convolution layer 
    conv_layer = Conv2D(n_filters, 
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer, 
                        **kwargs)(input_layer)
    
    # Apply chosen normalization method
    if batch_normalization:
        # Apply Batch normalization layer 
        norm_layer = BatchNormalization()(conv_layer)
    elif instance_normalization:
        # Apply Instance normalization
        norm_layer = InstanceNormalization()(conv_layer)

    # Activation function
    layer = activation(norm_layer) 
    
    return layer



def context_module(input, n_filters):
    """
    The activations in the context pathway are computed by context modules.
    Each context module is a pre-activation residual block with two
    3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between.
    """
    conv1 = conv2D_layer(input, n_filters, kernel_size=(3, 3), strides=(1, 1), activation='LeakyReLU', instance_normalization=True)
    dropout = Dropout(rate=0.3)(conv1)
    conv2 = conv2D_layer(dropout, n_filters, kernel_size=(3, 3), strides=(1, 1), activation='LeakyReLU', instance_normalization=True)

    return conv2

# improved unet model
def unet_model(output_channels, f=64):
    """
    Improved UNet network based on https://arxiv.org/abs/1802.10508v1.

    Comprises of a context aggregation pathway that encodes increasingly abstract 
    representations of the input as we progress deeper into the network, followed 
    by a localization pathway that recombines these representations with shallower 
    features to precisely localize the structures of interest.



    output_channels: Correspond to the classes a pixel can be. Here 4.
    f: Filters used in convolutional layers.
    
    Reference: https://arxiv.org/abs/1802.10508v1
    """
    inputs = tf.keras.layers.Input(shape=(image_pixel_rows, image_pixel_cols, image_channels))
    
    # Downsampling through the model
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(inputs)
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(d1)
    
    d2 = tf.keras.layers.MaxPooling2D()(d1)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    
    d3 = tf.keras.layers.MaxPooling2D()(d2)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    
    d4 = tf.keras.layers.MaxPooling2D()(d3)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    
    d5 = tf.keras.layers.MaxPooling2D()(d4)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    
    # Upsampling and establishing skip connections
    u4 = tf.keras.layers.UpSampling2D()(d5)
    u4 = tf.keras.layers.concatenate([u4, d4])
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    
    u3 = tf.keras.layers.UpSampling2D()(u4)
    u3 = tf.keras.layers.concatenate([u3, d3])
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    
    u2 = tf.keras.layers.UpSampling2D()(u3)
    u2 = tf.keras.layers.concatenate([u2, d2])
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    
    u1 = tf.keras.layers.UpSampling2D()(u2)
    u1 = tf.keras.layers.concatenate([u1, d1])
    u1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(u1)
    
    # This is the last layer of the model
    # Multiclass classification, to use softmax to output class predctions that
    # sum to 1 (i.e. probabilities). If problem involved binary classification of
    # pixels, the can use a sigmoid (0 or 1).
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)





# End of operation
print('End')
