"""
    This file contains the class "Trainer", which handles training the GCN model.
"""

__author__ = "Lachlan Comino"


from modules import GCN_Model, GCN_Layer
import numpy as np
import os
from tensorflow.keras import models


EPOCHS = 3000    # number of epochs


class Trainer:
    """
        Keeps all the functions needed for training as well as saving/loading the model.
    """
    def __init__(self, epochs=EPOCHS, overwrite=False):
        """
            Initialises the trainer, as well as the model itself
            @Params:
                epochs: number of epochs to train with
                overwrite: overwrite the already saved model if True
        """
        self.epochs = epochs

        # Initialise model
        self.gcn = GCN_Model()
        self.model_dir = 'recognition/45321199-task#2-gcn/saved_model/saved_gcn.h5'
        self.history_dir = 'recognition/45321199-task#2-gcn/saved_model/history.npy'
        self.overwrite = overwrite
    
    def train(self):
        """
            Trains the model and saves it as well as the history.
        """
        # Train
        if not os.path.exists(self.model_dir) or self.overwrite == True:
            data = self.gcn.data
            history = self.gcn.model.fit(data['validation_data'][0], 
                                    data['encoded_labels'],
                                    sample_weight=data['train_mask'],
                                    epochs=self.epochs,
                                    batch_size=data['len_vertices'],
                                    validation_data=data['validation_data'],
                                    shuffle=False)
            # save model
            self.gcn.model.save(self.model_dir)

            #save history
            np.save(self.history_dir, history.history)

    def get_model(self):
        """
            Load the model.
        """
        return models.load_model(self.model_dir, 
                                custom_objects={"GCN_Layer": GCN_Layer})

    def get_history(self):
        """
            Load the training history
        """
        return np.load(self.history_dir, allow_pickle='TRUE').item()
    
    def get_model_dir(self):
        """
            returns the directory of the saved model.
        """
        return self.model_dir