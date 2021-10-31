import torch.nn.functional as F


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