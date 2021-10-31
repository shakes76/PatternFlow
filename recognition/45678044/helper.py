import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import natsort
from PIL import Image
import os
import matplotlib.pyplot as plt
import pytorch_msssim.ssim as cal_ssim
from tqdm import tqdm

def train(model, optim, epoch_size, train_loader, valid_loader=None):
    train_loss = []
    model.train()
    
    for epoch in range(epoch_size):
        epoch_loss = []
        
        for batch, (imgs, _) in enumerate(train_loader):
            encoded, decoded, vq_loss = model(imgs)
          
            reconst_loss = F.mse_loss(decoded, imgs)
            
            loss = reconst_loss + vq_loss
          
            optim.zero_grad()
            loss.backward()
            optim.step() 
            
            epoch_loss.append(loss.detach().cpu().numpy())
      
        train_loss.append(np.mean(epoch_loss))
        print('Epoch: ', epoch + 1, '| Loss: ', train_loss[-1])
        
    return train_loss
    
    
def test(model, data_loader, optim, epoch_size):
    pass

def preload_imgs(path):
    imgs = os.listdir(path)

    transform = transforms.Compose([
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()
                                    ])
    
    imgs_tensor = []
    
    for i in range(len(imgs)):
        img_path = os.path.join(path, imgs[i])
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        imgs_tensor.append(img_tensor)
        print(i)
        
    dataset = torch.empty((len(imgs_tensor), 
                           imgs_tensor[0].shape[0],
                           imgs_tensor[0].shape[1],
                           imgs_tensor[0].shape[2],
                           ))
    
    for i in range(len(imgs_tensor)):
        dataset[i] = dataset[i] + imgs_tensor[i]
        
    return dataset

def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')