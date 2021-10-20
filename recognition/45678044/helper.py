import torch.nn.functional as F


def train(model, optim, epoch_size, train_loader, valid_loader=None):
    train_loss = []
    model.train()
    
    for epoch in range(epoch_size):
        epoch_loss = []
        
        for batch, (imgs, _) in enumerate(train_loader):
            z_e, encoded, decoded = model(imgs)
          
            reconst_loss = F.mse_loss(decoded, imgs)
            codebook_loss = F.mse_loss(encoded, z_e.detach())
            commit_loss = F.mse_loss(z_e, encoded.detach())
            loss = reconst_loss + codebook_loss + 0.25 * commit_loss
          
            optim.zero_grad()
            loss.backward()
            optim.step() 
            
            epoch_loss.append(loss.detach().cpu().numpy())
      
        train_loss.append(np.mean(epoch_loss))
        print('Epoch: ', epoch + 1, '| Loss: ', train_loss[-1])
        
    return train_loss
    
    
def test(model, data_loader, optim, epoch_size):
    pass