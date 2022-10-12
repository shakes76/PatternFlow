"""
Example usage of the GCN model applied to the Facebook dataset.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""
from dataset import Dataset
from train import GNNClassifier, loadClassifier

def main():
    path = "C:\\Users\\cyeol\\Documents\\University\\2022\\COMP3710\\facebook.npz"
    dataset = Dataset(path)
    dataset.summary(3)
    test_input = dataset.get_valid_split()

    trainer = loadClassifier(dataset, "C:\\Users\\cyeol\\Downloads\\test1")
    trainer.get_summary()
    trainer.predict_and_report()

    """epochs = 2
    batch_size = 256
    hidden_nodes = [32, 32]

    trainer = GNNClassifier(dataset, hidden_nodes)
    #trainer.get_summary()
    #history = trainer.train(epochs, batch_size)
    #trainer.plot_curves(history)

    trainer.save("C:\\Users\\cyeol\\Downloads\\test1")
    new_trainer = loadClassifier(dataset, "C:\\Users\\cyeol\\Downloads\\test1")
    new_trainer.get_summary()
    new_trainer.save("C:\\Users\\cyeol\\Downloads\\test2")
    another_trainer = loadClassifier(dataset, "C:\\Users\\cyeol\\Downloads\\test2")
    another_trainer.predict(dataset.get_valid_split())
    another_trainer.train(epochs, batch_size)"""

if __name__ == "__main__":
    main()