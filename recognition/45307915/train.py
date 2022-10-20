import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from math import floor

IMAGES_PATH = "./ISIC-2017_Training_Data/*.jpg"
MASKS_PATH = "./ISIC-2017_Training_Part1_GroundTruth/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

def preprocessImages(filenames):
    """
    Load and preprocess the image files.

        Parameters:
            filenames (tf.string): names of all image files

        Return:
            tf.Dataset: A (IMAGE_HEIGHT, IMAGE_WIDTH, 1) tensor containing all the image file data
    
    """
    raw = tf.io.read_file(filenames)
    images = tf.io.decode_jpeg(raw, channels=1)
    
    #resize images
    images = tf.image.resize(images, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    #Normalise
    images = images / 255.
    
    print(images)
    
    return images
    
    
def preprocessMasks(filenames):
    """
    Load and preprocess the mask files.

        Parameters:
            filenames (tf.string): names of all mask files

        Return:
            tf.Dataset: A (IMAGE_HEIGHT, IMAGE_WIDTH, 1) tensor containing all the mask file data
    
    """
    raw = tf.io.read_file(filenames)
    images = tf.io.decode_png(raw, channels=1)
    
    #resize images
    images = tf.image.resize(images, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    #Normalise
    images = images / 255.
    
    #Threshold image to 0-1
    images = tf.where(images > 0.5, 1.0, 0.0)
    
    return images

def loadData():
    """
    Loads and prepocesses all the image and mask data, located at IMAGES_PATH and MASKS_PATH.
         

        Return:
            tf.Dataset: A (IMAGE_HEIGHT, IMAGE_WIDTH, 1) tensor containing the processed image and mask data
    
    """
    
    image_data = tf.data.Dataset.list_files(IMAGES_PATH, shuffle=False)
    processedImages = image_data.map(preprocessImages)
    
    masks_data = tf.data.Dataset.list_files(MASKS_PATH, shuffle=False)
    processedMasks = masks_data.map(preprocessMasks)
    
    # Testing that pre-processing was successful
    for elem in processedImages.take(1):
        plt.imshow(elem.numpy())
        plt.show()
        
    for elem in processedMasks.take(1):
        plt.imshow(elem.numpy())
        plt.show()
    
    dataset = tf.data.Dataset.zip((processedImages, processedMasks))
    print(dataset)
    
    return dataset