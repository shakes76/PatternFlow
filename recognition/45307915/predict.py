import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

import modules
import dataset
import train

BATCH_SIZE = 32

class ModelPredictor():
    
    def __init__(self, model, test_dataset, batch_size=BATCH_SIZE):
        """
        ???????

        Parameters:
            ?????? (??????): ??????
            
        Return:
            ??????: ?????

        """
        
        self.model = model
        self.test_dataset = test_dataset
        self.test_batch = test_dataset.batch(batch_size)
        
    def evaluateModel(self):
        """
        ???????

        Parameters:
            ?????? (??????): ??????
            
        Return:
            ??????: ?????

        """
        
        coefficients = []
        for image, mask in self.test_batch:
            loss, coefficient = self.model.evaluateData(self.test_batch)
            coefficients.append(coefficients)
            
        averageCoefficient = sum(coefficients)/len(coefficients)
        print("Average Dice Coefficient: " + averageCoefficient)
        minCoefficient = min(coefficients)
        print("Minimum Coefficient: " + minCoefficient)
        
    def makePredictions(self):
        """
        ???????

        Parameters:
            ?????? (??????): ??????
            
        Return:
            ??????: ?????

        """
        
        batch_size = 10
        test_batch = self.test_dataset.batch(batch_size)
        test_image, test_mask = next(iter(test_batch))
        predictions = self.model.predict(test_image)
        
        for i in range(batch_size):
            
            plt.figure(figsize=(10,10))
        
            # Plot the test image
            plt.subplot(1, 3, 1)
            plt.imshow(test_image[i])
            plt.title("Input")

            # Plot the test mask
            plt.subplot(1, 3, 2)
            plt.imshow(test_mask[i])
            plt.title("Ground Truth Mask")

            # Plot the resultant mask
            plt.subplot(1, 3, 3)
            # Display 0 or 1 for classes
            prediction = tf.where(predictions[i] > 0.5, 1.0, 0.0)
            plt.imshow(prediction)
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