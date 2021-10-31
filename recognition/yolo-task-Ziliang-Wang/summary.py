import torch
from torchsummary import summary
from model.model import YoloBody
import driver

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  summary(YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], len(driver.class_names)).to(device), input_size=(3, 416, 416))
