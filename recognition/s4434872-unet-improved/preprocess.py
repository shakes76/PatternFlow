"""
Dataset preprocessing module for OASIS Brain Dataset.

@author Dhilan Singh (44348724)

Created: 06/11/2020
"""
import tensorflow as tf

def decode_png(file_path):
    """
    Decode raw data from filenames into png image with
    the desired channels and size.

    @param: file_path:
        The full path of the image file.
    @returns:
        png image with 1 channel (grayscale) and size 256x256.

    Reference: Adapted from Siyu lu
    """
    # Load the raw data from the file as a string
    png = tf.io.read_file(file_path)
    # Convert the compressed string to a uint8 tensor.
    # channels=3 for RGB, channels=1 for grayscale
    png = tf.image.decode_png(png, channels=1)
    # Resize the image to the desired size (size of network).
    png = tf.image.resize(png, (256, 256))
    return png

def process_path(image_fp, mask_fp):
    """
    Preprocess an image and segmentation mask pair from file paths.
    Normalizes the images and pixel-wise one-hot encodes the 
    segmentation masks.

    @param: image_fp:
       Path to the image file.
    @param: mask_fp:
        Path to the mask file.
    @returns:
        Normalized image (256x256x1) and one-hot encoded segmentation mask,
        (256x256x4) as png images.
    """
    # Get image as a tensor with channels
    image = decode_png(image_fp)
    # Standardise values to be in the [0, 1] range (only for images) (IMPORTANT).
    # If were using a GAN, then would make sense to normalise the target images as well.
    image = tf.cast(image, tf.float32) / 255.0
    
    # Get mask as a tensor with channels
    mask = decode_png(mask_fp)
    # One-hot encode each pixel of the mask.
    # From np.unique(mask) get => [0, 85, 170, 255], which are
    # the pixel classes of the segmentation. We want to map these
    # from these pixel values to our class values (1, 2, 3, 4), i.e.
    # mask[mask==0] = 0 (Background)
    # mask[mask==85] = 1 (CSF)
    # mask[mask==170] = 2 (Gray matter)
    # mask[mask=255] = 3 (White matter)
    mask = mask == [0, 85, 170, 255]
    
    return image, mask