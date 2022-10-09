"""
The training procedure to fit the GCN to the Facebook dataset.

Includes source code for training, validating, testing and saving the model. Also includes loss plots and metrics.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from modules import GNN
from dataset import Dataset

class GNNTrainer():

    def __init__(self, data: Dataset, epochs, batch_size, hidden_nodes, learning_rate=0.01, dropout_rate=0.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = len(set(data.get_targets()))

        # Construct model
        self.graph = data
        self.model = GNN(self.graph, self.num_classes, hidden_nodes, aggregation_type="sum", dropout_rate=dropout_rate)
        self.model.get_summary()