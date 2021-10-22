import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import model
import os
from model import *

## References
# list_files, zip, take, skip: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# io code: https://www.tensorflow.org/api_docs/python/tf/io/read_file
# image resizing: https://www.tensorflow.org/api_docs/python/tf/image/resize
# jpeg decoding: https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

# Data loading variables
ISIC_DATA = "./ISIC2018_Task1-2_Training_Input_x2/*.jpg" 
ISIC_MASKS = "./ISIC2018_Task1_Training_GroundTruth_x2/*.png"
PREPROCESS_STATUS = "DATA"

# Data managing variables
BATCH_SIZE = 32
DATASET_SIZE = 2594


def preprocess(filenames):
    raw_data = tf.io.read_file(filenames)
    raw_image = []
    
    if PREPROCESS_STATUS == "DATA":
        raw_image = tf.io.decode_jpeg(raw_data, channels=3)
    else:
        raw_image = tf.image.decode_png(raw_data, channels=1)
        
    # Resize the images
    raw_image = tf.image.resize(raw_image, [256,256])
    
    # Normalise
    raw_image = raw_image / 255.0
    return raw_image
    
    

def loadData():
    print("Function handling the loading, preprocessing and returning of ready-to-use data.")
    # Get the dataset contents
    isics_data = tf.data.Dataset.list_files(ISIC_DATA)
    processedData = isics_data.map(preprocess)
    print("Finished processing ISICs data...")
    
    # Get the corresponding segmentation masks
    PREPROCESS_STATUS = "MASKS"
    masks_data = tf.data.Dataset.list_files(ISIC_MASKS)
    processedMasks = masks_data.map(preprocess)
    print("Finished processing ISICs masks...")
    
    
    # Testing that pre-processing was successful
    for elem in processedData.take(1):
        plt.imshow(elem.numpy())
        plt.show()
        
    for elem in processedMasks.take(1):
        plt.imshow(elem.numpy())
        plt.show()
    
    # Return the newly created dataset
    return tf.data.Dataset.zip((processedData, processedMasks))


def splitData(dataset):
    print("Function handling the splitting and batching of data.")
    
    # Define the sizes for a 70 / 15 / 15 split
    training_size = int(0.7 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)
    validation_size = test_size
    
    # Use skip() and take() to split the data up
    training_set = dataset.take(training_size)
    
    # Training data is used up now
    dataset = dataset.skip(training_size)
    
    # Split the rest between the testing and validation
    testing_set = dataset.take(test_size)
    validation_set = dataset.skip(test_size)
    
    # Perform batching
    training_set.batch(BATCH_SIZE)
    testing_set.batch(BATCH_SIZE)
    validation_set.batch(BATCH_SIZE)
    
    return training_set, testing_set, validation_set
    

def main():
    # Dependencies
    print("Tensorflow: " + tf.__version__)
    print("Matplotlib: " + mpl.__version__)
    
    # Data loading and processing
    entire_dataset = loadData()
    train_data, test_data, validation_data = splitData(entire_dataset)
    
    # Create the model
    iunet = IUNET()
    model = iunet.createPipeline()
    


if __name__ == "__main__":
    main()
