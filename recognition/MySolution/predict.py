import numpy as np


"""
Outputs the overall accuracy of the data based on the model trained 
"""
class Predict:
    """
    Constructor of Predict class used to call plot_data()
    """
    def __init__(self, model):
        self.plot_data(model)

    """
    Uses the test data to return the accuracy of the model.
    """
    def plot_data(self, model):
        test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(np.load('X_test.npy'),  np.load('y_test.npy'),  verbose=2)
        print(f"test_loss: {test_loss}")
        print(f"test_acc: {test_acc}")
