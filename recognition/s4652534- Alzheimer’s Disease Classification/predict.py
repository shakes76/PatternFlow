
import numpy as np
import torch
from dataset import ADNI
from modules import resnet34
from utils import parse_data
from matplotlib import pyplot as plt



if __name__ == '__main__':

    height = 224
    width = 224
    num_features = 128
    num_classes = 2
    batch_size = 15
    dataset_dir = "./AD_NC"

    dataset = ADNI(dataset_dir)
    test_loader = dataset.get_test_loader(height, width, batch_size)
    label_dict = {0: "NC", 1: "AD"}

    model = resnet34(num_features=num_features, num_classes=num_classes, pretraining=False)
    model.cuda()
    model.load_state_dict(torch.load("best_model.pkl"))
    model.eval()

    inputs = next(iter(test_loader))
    imgs, labels, indexes = parse_data(inputs)
    emds, probs = model(imgs)
    _, predicted = torch.max(probs, 1)

    imgs = imgs.cpu().numpy().transpose(0,2,3,1)
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    fig = plt.figure(figsize=(12, 8))

    for i in range(15):
        subplot = fig.add_subplot(3, 5, i+1)
        subplot.imshow(imgs[i])
        subplot.title.set_text(
            "GT/Pre:" + label_dict[labels[i]] + "/" + label_dict[predicted[i]])

    plt.savefig("demo.jpg")




