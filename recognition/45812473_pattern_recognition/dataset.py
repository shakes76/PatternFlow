import os
import cv2
import tensorflow as tf
import numpy as np
from tf.keras.utils import to_categorical

def process_data(data_path, masks_path, image_size):
    """
    Processes all the data
    Takes the path to data and path to masks
    Returns a list of the data in an array and a corresponding list of the masks in an array
    """
    # Store all the images and masks in lists
    data = []
    masks = []

    for file in os.listdir(data_path):

        # For all files ending with jpg
        if not file.endswith(".jpg"):
            continue

        # Get the original data image and the corresponding mask image in greyscale
        data_image = cv2.imread(os.path.join(data_path, file), 0)
        mask_image = cv2.imread(os.path.join(masks_path, file.replace(".jpg", "_Segmentation.png")), 0)

        # Resize images
        data_image = cv2.resize(data_image, image_size)
        mask_image = cv2.resize(mask_image, image_size)

        # Normalize 
        data_image = data_image / 255.0
        mask_image = mask_image / 255.0
        data.append(data_image)
        masks.append(mask_image)
    
    # Convert to numpy arrays
    data = np.array(data)
    masks = np.array(masks)
    # One-hot encoding for masks
    masks = to_categorical(masks, num_classes = 2)
    
    return data, masks
