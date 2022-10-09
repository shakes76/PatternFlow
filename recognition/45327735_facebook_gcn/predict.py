"""
Example usage of the GCN model applied to the Facebook dataset.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""
from dataset import Dataset
from train import GNNTrainer

def main():
    path = "C:\\Users\\cyeol\\Documents\\University\\2022\\COMP3710"
    filename = "facebook"
    dataset = Dataset(path, filename)
    dataset.summary(3)

    epochs = 3
    batch_size = 256
    hidden_nodes = [32, 32]

    trainer = GNNTrainer(dataset, hidden_nodes)
    trainer.get_summary()
    history = trainer.train(epochs, batch_size)
    trainer.plot_curves(history)

if __name__ == "__main__":
    main()