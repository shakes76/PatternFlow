# This is the test driver script.
# This script shows example usage of the module created in solution.py
# It creates relevant plots and visualisations


# include a main method
# can run the solution

# can use numpy if needed.

# want to show an example of my solution being run.

# import the correct modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import solution


def import_data():
    
    return 0


def manipulate_data(data):
    
    return 0


def plot():
    """
    Plot the images
    """
    
    return 0


def analyse_training_history(history):
    
    # analyse the model
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    return 0


def analyse_predictions(predictions, X_test, Y_test, target_names):
    
    # determine if predictions were correct
    correct = (predictions == Y_test)

    # number of images testes
    total_test = len(X_test)

    print("Total Tests:", total_test)
    print("Predictions:", predictions)
    print("Which Correct:", correct)
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", np.sum(correct)/total_test)

    print(classification_report(Y_test, predictions, target_names=target_names))
    
    return 0


def main():
    print("Running the driver script")
    return 0
    
    
    # import the data
    data = import_data()
    
    # manipulate the data
    X_train, X_test, Y_train, Y_test, target_names, h, w, n_classes = manipulate_data(data)
    
    # plot example images
    
    
    # create the model
    model = solution(h, w, n_classes)
    
    # train the model
    history = model.fit(X_train, Y_train, epochs=100, validation_split=0.3)
    
    # analyse history of training the model
    analyse_training_history(history)
    
    # make predictions using the model
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    
    # show plots of some of these predictions
    
    
    # numerically analyse the performance model
    analyse_predictions(predictions, X_test, Y_test, target_names)
    
    return 0
    
    

if __name__ == "__main__":
    main()