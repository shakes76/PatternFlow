from modules import GCN_Model


EPOCHS = 1000     # number of epochs


class Trainer:
    def __init__(self, epochs=EPOCHS):
        self.epochs = epochs

        self.gcn = GCN_Model()
        
    
    def train(self):
        # Train
        data = self.gcn.data
        self.history = self.gcn.model.fit(data['validation_data'][0], 
                                data['encoded_labels'],
                                sample_weight=data['train_mask'],
                                epochs=self.epochs,
                                batch_size=data['len_vertices'],
                                validation_data=data['validation_data'],
                                shuffle=False)

    def get_history(self):
        return self.history