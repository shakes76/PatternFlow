import torch
from matplotlib import pyplot as plt
import modules
from dataset import Dataset
from train import dsc, get_mask

if __name__ == "__main__":
    ## set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    batch_size = 1
    dataset_test = Dataset(
        data_path='data/test/data', 
        truth_path='data/test/truth', 
        metadata_path='data/test/data/ISIC-2017_Test_v2_Data_metadata.csv'
        )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = modules.IUNET(3, 16)
    model.load_state_dict(torch.load('model.pt'))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # evaluate model performance metrics
        losses = []
        dscs = []
        for batch_no, (img, mask) in enumerate(dataloader_test):
            img = img.to(device)
            mask = mask.to(device)

            out = model(img)
            loss = 1 - dsc(out[:,0,:,:], mask[:,0,:,:])
            losses.append(loss)

            pred = get_mask(out[:,0,:,:], 0.5)
            dscs.append(dsc(pred, mask[:,0,:,:]))
        avg_loss = sum(losses)/len(losses)
        avg_dsc = sum(dscs)/len(dscs)

        print(f"loss: {avg_loss:.2f}\tdsc: {avg_dsc:.2f}")
        
        # display first 4 predicted masks
        fig = plt.figure()
        loader = iter(dataloader_test)
        for i in range(4):
            img, mask = next(loader)
            img = img.to(device)

            out = model(img)
            out = out.to('cpu')[0].permute(1,2,0)[:,:,0]
            pred = torch.where(out >= 0.5, 1.0, 0.0)
            
            fig.add_subplot(4, 4, 1 + 4*i)
            plt.imshow(img.to('cpu')[0].permute(1,2,0))
            fig.add_subplot(4, 4, 2 + 4*i)
            plt.imshow(mask.to('cpu')[0].permute(1,2,0))
            fig.add_subplot(4, 4, 3 + 4*i)
            plt.imshow(out)
            fig.add_subplot(4, 4, 4 + 4*i)
            plt.imshow(pred)
        plt.show()
        plt.axis('off')