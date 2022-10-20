import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor

IMAGES_PATH = "./ISIC-2017_Training_Data/*.jpg"
MASKS_PATH = "./ISIC-2017_Training_Part1_GroundTruth/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

class DataLoader():
    
    def __init__(self, images_path=IMAGES_PATH, masks_path=MASKS_PATH, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        """ Create a new Yolo Model instance.
        
        Parameters:
            images_path (str): Path of the dataset images
            masks_path (str):  Path of the dataset masks
            img_width (int): Image Width
            img_height (int): Image Height
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_width = img_width
        self.img_height = img_height
        
        self.data = self.loadData()
        
    def preprocessImages(self, filenames):
        """
        Load and preprocess the image files.

            Parameters:
                filenames (tf.string): names of all image files

            Return:
                tf.Dataset: A (img_height, img_width, 1) tensor containing all the image file data

        """
        raw = tf.io.read_file(filenames)
        images = tf.io.decode_jpeg(raw, channels=1)

        #resize images
        images = tf.image.resize(images, [self.img_height, self.img_width])

        #Normalise
        images = images / 255.

        print(images)

        return images

    def preprocessMasks(self, filenames):
        """
        Load and preprocess the mask files.

            Parameters:
                filenames (tf.string): names of all mask files

            Return:
                tf.Dataset: A (img_height, img_width, 1) tensor containing all the mask file data

        """
        raw = tf.io.read_file(filenames)
        images = tf.io.decode_png(raw, channels=1)

        #resize images
        images = tf.image.resize(images, [self.img_height, self.img_width])

        #Normalise
        images = images / 255.

        #Threshold image to 0-1
        images = tf.where(images > 0.5, 1.0, 0.0)

        return images
    
    def loadData(self):
        """
        Loads and prepocesses all the image and mask data, located at IMAGES_PATH and MASKS_PATH.


            Return:
                tf.Dataset: A (img_height, img_width, 1) tensor containing the processed image and mask data

        """

        image_data = tf.data.Dataset.list_files(self.images_path, shuffle=False)
        processedImages = image_data.map(preprocessImages)

        masks_data = tf.data.Dataset.list_files(self.masks_path, shuffle=False)
        processedMasks = masks_data.map(preprocessMasks)

        dataset = tf.data.Dataset.zip((processedImages, processedMasks))

        return dataset