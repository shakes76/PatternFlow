import torch
from numpy import random
from dataset import ADNIDataset

def predict():
    device = torch.device("cpu")
    ds = ADNIDataset("/home/jingweini/Documents/uni/COMP3710/report/data/AD_NC", device)
    img_seq, label = ds[random.randint(0, ADNIDataset.NUM_SCAN_GROUPS)]
    # print(img_seq.shape, label.shape)
    print(label.item())
    model = torch.load("model_0.1.pkl").to(device)
    out = model(img_seq.unsqueeze(0))
    print(out.item())

if __name__ == "__main__":
    predict()
