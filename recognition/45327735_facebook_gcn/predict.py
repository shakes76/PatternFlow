"""
Example usage of the GCN model applied to the Facebook dataset.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""
from dataset import Dataset
from train import GNNClassifier, loadClassifier

"""
Shows example usage of the algorithm - including constructing the Dataset, constructing the classifier, plotting figures
and loading saved models.  

NOTE: The user must manually change the 'path' variable to point to the facebook.npz file.
"""
def main():
    # Create dataset
    path = "C:\\insert\\path\\to\\facebook.npz\\data\\here"
    dataset = Dataset(path)

    # Initialise
    epochs = 10
    save_path = ".\\saved_model"

    print("CONSTRUCT A CLASSIFIER . . . ")
    trainer = GNNClassifier(dataset, epochs=epochs, save_path=save_path)
    trainer.get_summary()
    trainer.predict_and_report()
    trainer.plot_curves()
    trainer.plot_umap()

    print("LOAD A SAVED CLASSIFIER . . . ")
    trainer = loadClassifier(dataset, save_path)
    trainer.get_summary()
    trainer.predict_and_report()


if __name__ == "__main__":
    main()
