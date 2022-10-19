from train import Trainer
import matplotlib.pyplot as plt
import os


class Predicter:
    def __init__(self):
        self.trainer = Trainer()
        self.trainer.train()
        self.model_dir = self.trainer.model_dir
        self.figs_dir = 'recognition/45321199-task#2-gcn/figures/'

        # make figures dir
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)


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
