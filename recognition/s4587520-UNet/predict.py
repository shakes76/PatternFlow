from modules import UNet, dice_similarity
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToTensor

predict_image_path = "./data/test/image/ISIC_0000000.jpg"

#Import GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load Trained model and test dataset
model = UNet().to(device)
save_path = "./Trained_Model.pth"
model.load_state_dict(torch.load(save_path))

#Image Pathway
transform = ToTensor()
image = transform(Image.open(predict_image_path))

#Test model and save output
image = image.to(device)
output = model(image[None,:,:,:])
print(f"DICE Similarity = {dice_similarity(output, image[None,:,:,:])}")

#Show Output
plt.figure()
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(image.cpu().permute(1,2,0))
axarr[1].imshow(output[0,:,:,:].permute(1,2,0)[:,:,0].cpu().detach().numpy())
plt.show()