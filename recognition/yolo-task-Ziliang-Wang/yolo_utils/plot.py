import matplotlib.pyplot as plt
import os


class Plot_loss():
    def __init__(self, loss, test_loss, epoch):
        self.loss = loss
        self.test_loss = test_loss
        self.epoch = epoch
        self.save_path = "results/"

    def plot(self):
        plt.figure(figsize=(6, 4), dpi=200)
        plt.plot(self.loss, label="train_loss", color='#3D0FF2')
        plt.plot(self.test_loss, label="test_loss", linestyle='--', color='#EB935E')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("ISIC dataset train and test loss with adam optimizer")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "loss" + str(self.test_loss[-1]) + ".png"))
        plt.show()
