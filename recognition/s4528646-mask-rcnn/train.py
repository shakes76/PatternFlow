from dataset import *
from modules import model
import matplotlib.pyplot as plt
import torch

NUM_EPOCHS = 10
print("GPU Available:", torch.cuda.is_available())

train_data = ISICDataset("./data/ISIC-2017_Training_Data", "./data/ISIC-2017_Training_Part1_GroundTruth", "./data/ISIC-2017_Training_Part3_GroundTruth.csv")
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

model.eval()
model.float()
for i, batch in enumerate(train_data):
    image, label, mask = batch
    # plt.imshow(prediction[0]["masks"].detach().numpy().reshape(767, 1022, 3))
    print(mask.shape)
    if i > 10:
        break

