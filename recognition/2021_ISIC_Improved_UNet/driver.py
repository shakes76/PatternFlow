import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import model
import os
from model import *

## References
# list_files, zip, take, skip: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# io code: https://www.tensorflow.org/api_docs/python/tf/io/read_file
# image resizing: https://www.tensorflow.org/api_docs/python/tf/image/resize
# jpeg decoding: https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg
# image thresholding: https://www.tensorflow.org/api_docs/python/tf/where
# optimizer: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
# dice coefficient: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
# input dataset to fit(): https://www.tensorflow.org/guide/data

# Data loading variables
ISIC_DATA = "./ISIC2018_Task1-2_Training_Input_x2/*.jpg" 
ISIC_MASKS = "./ISIC2018_Task1_Training_GroundTruth_x2/*.png"

# Data managing variables
BATCH_SIZE = 32
DATASET_SIZE = 2594
EPOCHS = 10


def preprocessData(filenames):
    raw_data = tf.io.read_file(filenames)
    
    # Decode images
    raw_image = tf.io.decode_jpeg(raw_data, channels=3)
        
    # Resize the images
    raw_image = tf.image.resize(raw_image, [256,256])
    
    # Normalise
    raw_image = raw_image / 255.0
    return raw_image
    
    
def preprocessMasks(filenames):
    raw_data = tf.io.read_file(filenames)
    
    # Decode images
    raw_image = tf.io.decode_png(raw_data, channels=1)
        
    # Resize the images
    raw_image = tf.image.resize(raw_image, [256,256])
    
    # Normalise
    raw_image = raw_image / 255.0
    
    # Threshold image to 0-1
    raw_image = tf.where(raw_image > 0.5, 1.0, 0.0)
    return raw_image
    
    
def loadData():
    print("Function handling the loading, preprocessing and returning of ready-to-use data.")
    # Get the dataset contents
    isics_data = tf.data.Dataset.list_files(ISIC_DATA)
    processedData = isics_data.map(preprocessData)
    print("Finished processing ISICs data...")
    
    # Get the corresponding segmentation masks
    masks_data = tf.data.Dataset.list_files(ISIC_MASKS)
    processedMasks = masks_data.map(preprocessMasks)
    print("Finished processing ISICs masks...")
    
    
    # Testing that pre-processing was successful
    for elem in processedData.take(1):
        print(elem)
        print(elem.shape)
        plt.imshow(elem.numpy())
        plt.show()
        
    for elem in processedMasks.take(1):
        print(elem)
        print(elem.shape)
        print(np.unique(elem.numpy()))
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
    training_set = training_set.batch(BATCH_SIZE)
    testing_set = testing_set.batch(BATCH_SIZE)
    validation_set = validation_set.batch(BATCH_SIZE)
    
    return training_set, testing_set, validation_set
    
    
def diceCoefficient(y_true, y_pred):
    """  
        Defines the dice coefficient.
        
        The dice coefficient is defined as:
            2 * (Pixel Overlap)
         -----------------------------
          Total pixels in both images
          
        @param y_true: the true output
        @param y_pred: the output predicted by the model
        @return the dice coefficient for the prediction, based on the true output.
    """
    overlap = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    
    totalPixels = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) +  tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    
    diceCoeff = tf.keras.backend.mean((2.0 * overlap + 1) / (totalPixels + 1), axis=0)
    
    return diceCoeff
    
    
def diceLoss(y_true, y_pred):
    """
        Defines the dice coefficient loss function, ie 1 - Dice Coefficient.
        
        @param y_true: the true output
        @param y_pred: the output predicted by the model
        @return the dice coefficient subtracted from one. This allows dice similarity 
                to be used as a loss function.
    """
    return 1 - diceCoefficient(y_true, y_pred)


def main():
    # Dependencies
    print("Tensorflow: " + tf.__version__)
    print("Matplotlib: " + mpl.__version__)
    print("Numpy: " + np.__version__)
    
    # Data loading and processing
    entire_dataset = loadData()
    train_data, test_data, validation_data = splitData(entire_dataset)
    
    # Create the model
    iunet = IUNET()
    model = iunet.createPipeline()
    
    # Compile the model, model.compile()
    print("Compiling model...")
    adamOptimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=adamOptimizer, loss=diceLoss, metrics=[diceCoefficient, 'accuracy'])
    print("Model compilation complete.")
    
    # Train the model, model.fit()
    print("Training the model...")
    history = model.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=validation_data)
    print("Model training complete.")
    
    # Perform predictions, model.predict()
    


if __name__ == "__main__":
    main()
