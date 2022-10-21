import matplotlib.pyplot as plt
from PIL import Image
from modules.py import UNet

#Import GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load Trained model and test dataset
model = UNet().to(device)
save_path = "./Trained_Model.pth"
model.load_state_dict(torch.load(save_path))

#Image Pathway
predict_image_path = "./test_lesion"
transform = ToTensor()
image = transform(Image.open(predict_image_path))

#Test model and save output
image = image.to(device)
output = model(image[None,:,:,:])

#Show Output
plt.imshow(image.permute(1,2,0))
plt.show()
plt.imshow(output[0,:,:,:].permute(1,2,0)[:,:,0].detach().numpy())
plt.show()