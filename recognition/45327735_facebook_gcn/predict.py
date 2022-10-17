"""
Example usage of the GCN model applied to the Facebook dataset.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""
from dataset import Dataset
from train import GNNClassifier, loadClassifier


def main():
    # Create dataset
    path = "C:\\Users\\cyeol\\Documents\\University\\2022\\COMP3710\\facebook.npz"
    dataset = Dataset(path)
    dataset.summary(3)
    test_input = dataset.get_valid_split()

    # Initialise
    epochs = 1
    batch_size = 256
    hidden_nodes = [32, 32]
    save_path = "C:\\Users\\cyeol\\Downloads\\test1"

    print("CONSTRUCT A CLASSIFIER . . . ")
    trainer = GNNClassifier(dataset, epochs=epochs, batch_size=batch_size, hidden_nodes=hidden_nodes,
                            save_path=save_path)
    trainer.get_summary()
    trainer.predict_and_report()
    trainer.plot_curves()
    trainer.plot_umap()

    print("LOAD A SAVED CLASSIFIER . . . ")
    trainer = loadClassifier(dataset, "C:\\Users\\cyeol\\Downloads\\test1")
    trainer.predict_and_report()
    trainer.get_summary()
    trainer.plot_curves()
    trainer.plot_umap()


if __name__ == "__main__":
    main()
