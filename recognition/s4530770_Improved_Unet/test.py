from improved_unet import build_model
from data_loader import DataLoader

model = build_model((384, 512, 3))
model.summary()

#data = DataLoader("H:\\COMP3710\\ISIC2018_Task1-2_Training_Data\\")
#data.show_images()