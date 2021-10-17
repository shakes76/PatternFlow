"""
    Main driver script for the
    improved U-Net model.
    
    author: Richard Marron
    status: Development
"""
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalise_images(image_set):
    """
    Normalise images between 0 and 1
        Params:
            image_set : Numpy array of images
            
        Return : Normalised version of image_set
    """
    return image_set / 255.0

def load_images(path: str, ground_truth: bool=False, truncate: bool=False):
    """
    Get all images from a given path
        Params:
            path : Path to the dataset
            ground_truth : Whether the folder is the ground truth data
                           (These have images of format PNG)
            truncate : Whether to use the full dataset or only 1/3
        
        Return : Numpy array of images which are also Numpy arrays
    """
    if ground_truth:
        # Images have PNG format
        path += f"*.png"
    else:
        # Images have JPG format
        path += f"*.jpg"
    
    print("Loading image paths...")
    img_paths = glob.glob(path)
    if truncate:
        # Only take the first 1/3 of the images
        img_paths = img_paths[:len(img_paths)//3]
    print("Successfully loaded paths!") 
    print("Converting images into numpy arrays...")
    # Read in images from path and return numpy array
    return np.array([plt.imread(path).astype(np.float32) for path in img_paths], dtype=object)

def main(debugging=False):
    """
    Main program entry
        Params:
            debugging : When True, the data is truncated to 1/3
                        of original size so program runs faster
    """
    # Load normalised images
    input_images = normalise_images(load_images(path="./Data/ISIC2018_Task1-2_Training_Input_x2/", 
                                                ground_truth=False, truncate=debugging))
    # Load ground truth segmentation
    gt_images = load_images(path="./Data/ISIC2018_Task1_Training_GroundTruth_x2/", 
                            ground_truth=True, truncate=debugging)
    
    # Print out some useful information
    print(f"Total input images: {input_images.shape}")
    print(f"Total ground truth images: {gt_images.shape}")
    print(f"Input image shape: {input_images[0].shape}")
    print(f"Ground truth image shape: {gt_images[0].shape}")
    # Show example of image and it's segmentation
    plt.imshow(input_images[0][:, :])
    plt.figure()
    plt.imshow(gt_images[0][:, :])
    plt.show()
    

if __name__ == "__main__":
    main(debugging=True)