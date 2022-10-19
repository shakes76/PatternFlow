from modules import GCN_Model
import os


EPOCHS = 10     # number of epochs


class Trainer:
    def __init__(self, epochs=EPOCHS, overwrite=False):
        self.epochs = epochs
        self.gcn = GCN_Model()
        self.model_dir = 'recognition/45321199-task#2-gcn/saved_model/saved_gcn.h5'
        self.overwrite = overwrite
    
    def train(self):
        # Train
        if not os.path.exists(self.model_dir) or self.overwrite == True:
            data = self.gcn.data
            self.history = self.gcn.model.fit(data['validation_data'][0], 
                                    data['encoded_labels'],
                                    sample_weight=data['train_mask'],
                                    epochs=self.epochs,
                                    batch_size=data['len_vertices'],
                                    validation_data=data['validation_data'],
                                    shuffle=False)
            self.gcn.model.save(self.model_dir)

    def get_history(self):
        return self.history