from train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report


class Predicter:
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.train()
        self.model_dir = self.trainer.model_dir
        self.figs_dir = 'recognition/45321199-task#2-gcn/figures/'

        # Get processed data
        self.data = self.trainer.gcn.data

        # make figures dir
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)

    def run_all(self):
        self.predict()
        self.tsne_plot()
        self.acc_loss_plots()
    
    def predict(self):
        model = self.trainer.get_model()
        data = self.data

        predictions = model.predict(data['validation_data'][0], batch_size=data['len_vertices'])

        true = np.argmax(data['encoded_labels'][data['test_mask']], axis=1)
        pred = np.argmax(predictions[data['test_mask']], axis=1)

        report = classification_report(true, pred)
        print(report)

    def tsne_plot(self):
        pass

    def acc_loss_plots(self):
        history = self.trainer.get_history()

        acc = history['acc']
        val_acc = history['val_acc']
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(len(acc)) # Get number of epochs
        epochs_lim = float(len(epochs))

        #------------------------------------------------
        # Plot training and validation accuracy per epoch
        #------------------------------------------------
        plt.title('Validation Accuracy Vs Epochs')
        plt.plot(epochs, acc, 'r', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        plt.ylabel("Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylim(0.0, 1.1)
        plt.xlim(0.0, epochs_lim)
        plt.legend()
        plt.savefig(self.figs_dir + 'acc.png')

        #------------------------------------------------
        # Plot training and validation loss per epoch
        #------------------------------------------------
        plt.figure()
        plt.title('Validation Loss Vs Epochs')
        plt.plot(epochs, loss, 'r', label="Training Loss")
        plt.plot(epochs, val_loss, 'b', label="Validation Loss")
        plt.ylabel("Validation Loss")
        plt.xlabel("Epochs")
        plt.ylim(0.0, 1.1)
        plt.xlim(0.0, epochs_lim)
        plt.legend()
        plt.savefig(self.figs_dir + 'loss.png')
