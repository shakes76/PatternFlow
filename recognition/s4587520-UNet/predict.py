import matplotlib.pyplot as plt
from dataset.py import ISIC_Dataset
from modules.py import UNet
#Load Trained model and test dataset
model = UNet()

#TODO, Replace pathways with test dataset
test_image_path = "./ISIC-2017_Training_Data"
test_segmentation_path = "./ISIC-2017_Training_Part1_GroundTruth"

test_dataset = ISIC_Dataset(test_image_path, test_segmentation_path)

#Test model and save output
idx = 0
img, seg = test_dataset[idx]
plt.imshow(img.permute(1,2,0))
plt.show()
result = model(img[None,:,:,:])
plt.imshow(result[0,:,:,:].permute(1,2,0)[:,:,0].detach().numpy())
plt.show()