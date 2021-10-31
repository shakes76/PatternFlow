import matplotlib.pyplot as plt
import os


class Plot_loss():
    def __init__(self, loss, test_loss, iou_list, epoch):
        self.loss = loss
        self.test_loss = test_loss
        self.epoch = epoch
        self.save_path = "results/plot/"
        self.iou_list = iou_list

    def plot_loss(self):
        plt.figure(figsize=(8, 4), dpi=200)
        plt.plot(self.loss, label="train_loss", color='#3D0FF2')
        plt.plot(self.test_loss, label="test_loss", linestyle='--', color='#EB935E')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("ISIC dataset training and test loss")
        plt.legend()

        plt.savefig(os.path.join(self.save_path, "loss" + str(self.test_loss[-1]) + ".png"))
        plt.show()

    def plot_iou(self):
        plt.figure(figsize=(8, 4), dpi=200)
        plt.plot(self.iou_list, label="IOU curve", color='green')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.title("ISIC test set IOU, epoch 142, 150, 178, 179, 180")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "iou " + str(self.iou_list[-1]) + ".png"))
        plt.show()
