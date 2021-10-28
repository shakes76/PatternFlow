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
# dice coefficient 2: https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
# input dataset to fit(): https://www.tensorflow.org/guide/data
# plotting predictions: https://www.tensorflow.org/tutorials/load_data/images

# Data loading and processing variables
ISIC_DATA = "./ISIC2018_Task1-2_Training_Input_x2/*.jpg" 
ISIC_MASKS = "./ISIC2018_Task1_Training_GroundTruth_x2/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

# Data managing variables
BATCH_SIZE = 32
DATASET_SIZE = 2594

# Network variables
OPT_LEARNING_RATE = 5e-4
EPOCHS = 13


def preprocessData(filenames):
    """
        Loads and preprocesses the images. The images must be: 
            - decoded
            - reshaped to [IMAGE_HEIGHT, IMAGE_WIDTH] (chosen size)
            - normalised (pixels must be between 0 and 1)
            
        @param filenames: the names of all of the image files
        
        @return the newly processed images
    """
    raw_data = tf.io.read_file(filenames)
    
    # Decode images
    raw_image = tf.io.decode_jpeg(raw_data, channels=3)
        
    # Resize the images
    raw_image = tf.image.resize(raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    # Normalise
    raw_image = raw_image / 255.0
    
    return raw_image
    
    
def preprocessMasks(filenames):
    """
        Loads and preprocesses the masks. The masks must be: 
            - decoded and reduced to a single colour channel
            - reshaped to [IMAGE_HEIGHT, IMAGE_WIDTH] (chosen size)
            - normalised and thresholded (pixels must be 0 or 1)
            
        @param filenames: the names of all of the mask image files
        
        @return the newly processed masks
    """
    raw_data = tf.io.read_file(filenames)
    
    # Decode images
    raw_image = tf.io.decode_png(raw_data, channels=1)
        
    # Resize the images
    raw_image = tf.image.resize(raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    # Normalise
    raw_image = raw_image / 255.0
    
    # Threshold image to 0-1
    raw_image = tf.where(raw_image > 0.5, 1.0, 0.0)
    
    return raw_image
    
    
def loadData():
    print("Function handling the loading, preprocessing and returning of ready-to-use data.")
    # Get the dataset contents
    isics_data = tf.data.Dataset.list_files(ISIC_DATA, shuffle=False)
    processedData = isics_data.map(preprocessData)
    print("Finished processing ISICs data...")
    
    # Get the corresponding segmentation masks
    masks_data = tf.data.Dataset.list_files(ISIC_MASKS, shuffle=False)
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
    newDataSet = tf.data.Dataset.zip((processedData, processedMasks))

    return newDataSet


def splitData(dataset):
    """
        Splits the dataset into the 70 / 15 / 15 split. 
        
        @param dataset: the entire dataset
        
        @return the training, testing and validation datasets, now split 70 / 15 / 15
    """
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

    return training_set, testing_set, validation_set
    
    
def performBatching(train, test, validation):
    """
        Performs batching of the three data sets; training, testing and validation.
        The size of the batches is defined in the variable BATCH_SIZE
        
        @param train: the training dataset
        @param test: the testing dataset
        @param validation: the validation dataset
        
        @return the original datasets passed in, now batched.
    """
    training_set = train.batch(BATCH_SIZE)
    testing_set = test.batch(BATCH_SIZE)
    validation_set = validation.batch(BATCH_SIZE)
    
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
    #overlap = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    
    #totalPixels = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) +  tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    
    #diceCoeff = tf.keras.backend.mean((2.0 * overlap + 1) / (totalPixels + 1), axis=0)
    
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    diceCoeff = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.)
    
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


def generatePredictions(test_data, model):
    """
        Generates and displays predictions of the model from the test set.
        local_batch amount of figures are displayed, which are a single row with 3 images.
        The images are:
            1. The original image input to the network 
            2. The ground truth mask for that input 
            3. The resultant mask determined by the trained network.
        
        @param test_data: the unbatched test data for the network
        @param model: the network 
    """  
    local_batch = 20
    titles = ['Input image', 'Ground Truth Mask','Resultant Mask']
    test_data_batched = test_data.batch(local_batch)
    test_image, test_mask = next(iter(test_data_batched))  
    mask_prediction = model.predict(test_image)
    
    # Plot the original image, ground truth and result from the network.
    for i in range(local_batch):
        
        plt.figure(figsize=(10,10))
        
        # Plot the test image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image[i])
        plt.title("Input Image")
        plt.axis("off")
        
        # Plot the test mask
        plt.subplot(1, 3, 2)
        plt.imshow(test_mask[i])
        plt.title("Ground Truth Mask")
        plt.axis("off")
        
        # Plot the resultant mask
        plt.subplot(1, 3, 3)
        
        # Display 0 or 1 for classes
        prediction = tf.where(mask_prediction[i] > 0.5, 1.0, 0.0)
        plt.imshow(prediction)
        #print(np.unique(mask_prediction[i]))
        plt.title("Resultant Mask")
        plt.axis("off")
        
        plt.show()

    return 1


def plotHistory(history):
    """
        Plots the value vs epoch graphs for the Dice Coefficient and Dice 
        Coefficient loss throughout training and validation.
        
        @param history: the loss and coefficient history of the model 
                        throughout training.
    """
    print(history.history.keys())
    modelHistory = history.history
    
    # Loss plots
    plt.plot(modelHistory['loss'])
    plt.plot(modelHistory['val_loss'])
    plt.title('Dice Coefficient Loss')
    plt.ylabel('Loss (%)')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    # Accuracy plots
    plt.plot(modelHistory['diceCoefficient'])
    plt.plot(modelHistory['val_diceCoefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('DSC')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

def main():
    # Dependencies
    print("Tensorflow: " + tf.__version__)
    print("Matplotlib: " + mpl.__version__)
    print("Numpy: " + np.__version__)
    
    # Data loading and processing
    entire_dataset = loadData()
    train_data, test_data, validation_data = splitData(entire_dataset)
    
    train_data_batched, test_data_batched, validation_data_batched = performBatching(train_data, test_data, validation_data)
    
    # Create the model
    iunet = IUNET()
    model = iunet.createPipeline()
    
    # Compile the model, model.compile()
    print("Compiling model...")
    adamOptimizer = tf.keras.optimizers.Adam(learning_rate=OPT_LEARNING_RATE)
    model.compile(optimizer=adamOptimizer, loss=diceLoss, metrics=[diceCoefficient])
    print("Model compilation complete.")
    
    # Train the model, model.fit()
    print("Training the model...")
    history = model.fit(train_data_batched, epochs=EPOCHS, validation_data=validation_data_batched)
    print("Model training complete.")
    
    plotHistory(history)
    
    # Evaluate performance on test, model.evaluate()
    print("Evaluating the model on the test set...")
    i = 0
    lossV = []
    coefficientV = []
    under = 0
    fine = 0
    for test_image, test_mask in test_data.batch(1):
        loss, coefficient = model.evaluate(test_image, test_mask)
        lossV.append(loss)
        coefficientV.append(coefficient)
        
        if (coefficient < 0.8):
            under += 1
        else:
            fine += 1
        
        i += 1
    
    percentageFine = ((fine / i) * 100);
    averageDC = sum(coefficientV) / len(coefficientV)
    print(">>> Evaluating Test Set \n Test dataset size: " + str(i))
    print("Amount fine: " + str(fine))
    print("Amount under 0.8: " + str(under))
    print("Average Dice Coefficient: " + str(averageDC))
    print("---- " + str(percentageFine) + "% of Test Set has 0.8 Dice Coefficient or above ----")
    
    plt.hist(coefficientV)
    plt.title("Dice Coefficients of Test Set for Total Epochs: " + str(EPOCHS))
    plt.ylabel('Frequency')
    plt.xlabel('Dice Coefficient')
    plt.show()
        
 
    
    print("Model evaluation complete.")
    
    # Perform predictions, model.predict()
    print("Displaying model predictions...")
    generatePredictions(test_data, model)
    print("Model prediction complete.")
    


if __name__ == "__main__":
    main()
