import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import ImprovedUNETModel
from dataset import DataLoader
from train import ModelTrainer

BATCH_SIZE = 32

class ModelPredictor():
    
    def __init__(self, model, test_dataset, batch_size=BATCH_SIZE):
        """
        Create a new model predictor instance to make predictions using a trained model

        Parameters:
            model (tf.Model): Trained Improved UNET model
            test_datset (tf.Dataset): A dataset containing all the image and mask test data
        """
        
        self.model = model
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        
        
    def evaluateModel(self):
        """
        Evaluate the test dataset Dice Similarity of the model 

        Parameters:
            
        Return:
        """
        
        test_batch = self.test_dataset.batch(self.batch_size)
        loss, coefficient = self.model.evaluate(test_batch)

        print("Test Data Dice Coefficient: ", coefficient)
        
    def makePredictions(self):
        """
        Using the trained model make predictions on the testing dataset

        Parameters:
            
        Return:
        """
        
        test_batch = self.test_dataset.batch(self.batch_size)
        test_image, test_mask = next(iter(test_batch))
        predictions = self.model.predict(test_image)
        
        for i in range(self.batch_size):
            
            plt.figure(figsize=(10,10))
        
            # Plot the test image
            plt.subplot(1, 3, 1)
            plt.imshow(test_image[i])
            plt.title("Input")

            # Plot the test mask
            plt.subplot(1, 3, 2)
            mask = test_mask[i]
            plt.imshow(mask[:, :, 0], cmap='gray')
            plt.title("Ground Truth Mask")

            # Plot the resultant mask
            plt.subplot(1, 3, 3)
            # Display 0 or 1 for classes
            prediction = tf.where(predictions[i] > 0.5, 1.0, 0.0)
            plt.imshow(prediction[:, :, 0], cmap='gray')
            plt.title("Predicted Mask")

            plt.show()
            

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
    
    # Evaluate the model and make predictions
    p = ModelPredictor(model, test_dataset)
    testEval = p.evaluateModel()
    pred = p.makePredictions()

if __name__ == "__main__":
    main()