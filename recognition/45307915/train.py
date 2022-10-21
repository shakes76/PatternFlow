import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from modules import ImprovedUNETModel
from dataset import DataLoader

TRAIN_IMAGES_PATH = "./ISIC-2017_Training_Data/*.jpg"
TRAIN_MASKS_PATH = "./ISIC-2017_Training_Part1_GroundTruth/*.png"

TEST_IMAGES_PATH = "./ISIC-2017_Test_v2_Data/*.jpg"
TEST_MASKS_PATH = "./ISIC-2017_Test_v2_Part1_GroundTruth/*.png"

VALIDATE_IMAGES_PATH = "./ISIC-2017_Validation_Data/*.jpg"
VALIDATE_MASKS_PATH = "./ISIC-2017_Validation_Part1_GroundTruth/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

BATCH_SIZE = 16
INIT_LEARNING_RATE = 5e-4
EPOCHS = 20

class ModelTrainer():
    
    def __init__(self, batch_size=BATCH_SIZE, learning_rate=INIT_LEARNING_RATE, epochs=EPOCHS):
        """
        Create a new model trainer instance to train the model

        Parameters:
            batch_size (int): Batch size
            learning_rate (int): Learning rate for training
            epochs (int): Number of epochs for training
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def diceCoefficient(self, y_true, y_pred):
        """
        Dice Coefficient

        Parameters:
            y_true (tf.Tensor): true output
            y_true (tf.Tensor): output predicted by model

        Return:
            tf.Tensor: Dice coefficient based on true output and prediction

        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
        return dice
    
    def diceLoss(self, y_true, y_pred):
        """
        Dice loss function

        Parameters:
            y_true (tf.Tensor): true output
            y_true (tf.Tensor): output predicted by model

        Return:
            tf.Tensor: Dice coefficient to be used as a loss function 

        """
        return 1 - self.diceCoefficient(y_true, y_pred)

    def plotResults(self, history):
        """
        Plots Dice Coefficient and Dice Coefficient Loss vs Epoch.
        For both training and validation.

        Parameters:
            history (History): record of training and validation metrics

        """
        modelHistory = history.history

        #Loss plots
        plt.plot(modelHistory['loss'])
        plt.plot(modelHistory['val_loss'])
        plt.title('Dice Coefficient Loss')
        plt.ylabel('Loss (%)')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.show()

        #Accuracy plots
        plt.plot(modelHistory['diceCoefficient'])
        plt.plot(modelHistory['val_diceCoefficient'])
        plt.title('Dice Coefficient')
        plt.ylabel('DSC')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.show()
        
    def trainModel(self, train_dataset, test_dataset, validate_dataset, model):
        """
        Train the model using the training and validation datasets.
        Using Dice Coefficient as the loss function.

        Parameters:
            train_dataset (tf.Dataset): Dataset containing all the training data
            test_dataset (tf.Dataset): Dataset containing all the test data
            validate_dataset (tf.Dataset): Dataset containing all the validation data
            model (tf.model): Untrained improved UNET model
            
        Return:
            tf.Model: trained improved UNET model

        """
        
        # Batch the data
        train_batch = train_dataset.batch(self.batch_size)
        test_batch = test_dataset.batch(self.batch_size)
        validate_batch = validate_dataset.batch(self.batch_size)
        
        adamOptimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adamOptimizer, loss=self.diceLoss, metrics=[self.diceCoefficient])
        
        results = model.fit(train_batch, epochs=self.epochs, validation_data=validate_batch)
        
        self.plotResults(results)
        
        return model

def main():

    # Data loading and preprocessing
    dataLoader = DataLoader()
    train_dataset, test_dataset, validate_dataset = dataLoader.loadData()
    
    # Generate the model
    improvedUNETModel = ImprovedUNETModel()
    model = improvedUNETModel.modelArchitecture()
    
    # Train the model
    t = ModelTrainer()
    model = t.trainModel(train_dataset, test_dataset, validate_dataset, model)

if __name__ == "__main__":
    main()