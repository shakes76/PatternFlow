import matplotlib.pyplot as plt
import os


class Plot_loss():
    def __init__(self, loss, test_loss,avg_iou, epoch):
        self.loss = loss
        self.test_loss = test_loss
        self.epoch = epoch
        self.save_path = "results/"
        self.avg_iou_test = avg_iou
    def plot(self):
        plt.figure(figsize=(10, 4), dpi=200)
        plt.subplot(1,2,1)
        plt.plot(self.loss, label="train_loss", color='#3D0FF2')
        plt.plot(self.test_loss, label="test_loss", linestyle='--', color='#EB935E')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("ISIC dataset training and test loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.avg_iou_test, label="avg_iou_test", color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("ISIC dataset IOU on testset")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "loss" + str(self.test_loss[-1]) + ".png"))
        plt.show()
