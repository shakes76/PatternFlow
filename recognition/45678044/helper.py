import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import natsort
from PIL import Image
import os
import matplotlib.pyplot as plt
import pytorch_msssim.ssim as cal_ssim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optim, epoch_size, train_loader, valid_loader=None):
    train_status = {'total_loss': [], 'reconst_loss': [], 'vq_loss': []}
    
    for epoch in range(epoch_size):
        model.train()
        total = 0
        reconst = 0
        vq = 0
        
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader))
        train_loop.set_description(f"Epoch [{epoch+1}/{epoch_size}]")
        
        for batch, imgs in train_loop:
            imgs = imgs.to(device)
            
            encoded, decoded, vq_loss = model(imgs)
          
            reconst_loss = F.mse_loss(decoded, imgs)
            
            loss = reconst_loss + vq_loss
          
            optim.zero_grad()
            loss.backward()
            optim.step() 
            
            total += loss.item()
            reconst += reconst_loss.item()
            vq += vq_loss.item()
            
            train_loop.set_postfix(reconst_loss=reconst/(batch+1))
      
        train_status['total_loss'].append(total/(batch+1))
        train_status['reconst_loss'].append(reconst/(batch+1))
        train_status['vq_loss'].append(vq/(batch+1))
        
    return train_status
    
    
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