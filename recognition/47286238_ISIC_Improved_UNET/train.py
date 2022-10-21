from sched import scheduler
import torch, torchvision
from dataset import Dataset
from modules import IUNET
import matplotlib.pyplot as plt
import time
import pickle

def dsc(mask, truth):
    """
    Sørensen-Dice Similarity Coefficient
    Used as the loss function for the training loop
    Reference: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Formula

    mask: tensor of shape (N, 1, H, W), converted into a mask
    truth: tensor of shape (N, 1, H, W)
    """
    a = mask
    # b = torch.squeeze(truth, 1)
    b = truth

    # intersection = torch.where(a > 0.0, 0.0, b)
    intersection = a * b
    epsilon = 1e-8 # prevent division by zero error
    return 2 * torch.sum(intersection) / (torch.sum(a) + torch.sum(b) + epsilon)

def get_mask(predicted, threshold):
    """ 
    Convert model output into image mask
    Turns values at or above a given threshold into 1.0,
        below it becomes 0.0
    predicted: tensor of shape (N, 1, H, W)
    returns: tensor of shape (N, 1, H, W), adjusted to become a mask
    """
    mask = torch.where(predicted >= threshold, 1.0, 0.0)
    return mask

def evaluate_model(model, loader, limit, batch_size):
    """ 
    Report average loss and DSC values over validation dataset
    returns: Tuple(avg losses, avg DSC)
    """
    model.eval() # disable setting model weights
    with torch.no_grad():
        losses = []
        dscs = []
        for batch_no, (img, mask) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)

            out = model(img)
            loss = 1 - dsc(out[:,0,:,:], mask[:,0,:,:])
            losses.append(loss)

            pred = get_mask(out[:,0,:,:], 0.5)
            dscs.append(dsc(pred, mask[:,0,:,:]))

            if batch_no >= limit/batch_size:
                break
    model.train() # re-enable model learning
    return (sum(losses)/len(losses), sum(dscs)/len(dscs))


if __name__ == '__main__':
    start = time.time()

    ## set device
    if not torch.cuda.is_available():
        print("CUDA not available for training! Aborting...")
    device = 'cuda'

    # batch size set to 1 due to limited memory
    batch_size = 1

    # load datasets, see README.md for expected folder structure
    dataset_train = Dataset(
        data_path='data/training/data', 
        truth_path='data/training/truth', 
        metadata_path='data/training/data/ISIC-2017_Training_Data_metadata.csv'
        )
    dataset_validation = Dataset(
        data_path='data/validation/data', 
        truth_path='data/validation/truth', 
        metadata_path='data/validation/data/ISIC-2017_Validation_Data_metadata.csv'
        )

    # training loader set to shuffle in order to cover a wider set of the training set
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    

    # hyperparameters
    learning_rate = 5e-4
    decay = 0.985
    num_epochs = 15

    model = IUNET(3, 16).to(device)

    # set optimizer and learning rate schedule according to Isensee et al.
    # Adam optimizer + exponential learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    loss_history = []
    dsc_history = []
    validation_loss_history = []
    validation_dsc_history = []
    print('entering training loop')
    for i in range(num_epochs):
        epoch_start = time.time()
        ## train model
        losses = []
        dscs = []
        for batch_no, (img, mask) in enumerate(dataloader_train):
            img = img.to(device)
            mask = mask.to(device)

            # generate output
            out = model(img)

            # calculate dsc loss
            # loss = 1 - DSC(prediction, truth)
            loss = 1 - dsc(out[:,0,:,:], mask[:,0,:,:])
            losses.append(loss)

            # store dsc value
            pred = get_mask(out[:,0,:,:], 0.5)
            dscs.append(dsc(pred, mask[:,0,:,:]))

            # backpropagate
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # stop after 800 elements per epoch due to limited resources
            if batch_no == 800 / batch_size:
                break
        sched.step()

        epoch_loss = sum(losses)/len(losses)
        epoch_dsc = sum(dscs)/len(dscs)
        loss_history.append(epoch_loss)
        dsc_history.append(epoch_dsc)
        
        ## evaluate over validation dataset
        validation_loss, validation_dsc = evaluate_model(model, dataloader_validation, 100, 1)
        validation_loss_history.append(validation_loss)
        validation_dsc_history.append(validation_dsc)

        epoch_end = time.time()
        print(f"epoch [{i+1}/{num_epochs}] {(epoch_end - epoch_start)/60}m\tloss: {epoch_loss}\tdsc: {epoch_dsc}\tval loss: {validation_loss}\tval dsc: {validation_dsc}")

    # save files
    torch.save(model.state_dict(), 'model.pt')
    with open('train_loss.pkl', 'wb') as f:
        pickle.dump(loss_history, f)
    with open('train_dsc.pkl', 'wb') as f:
        pickle.dump(dsc_history, f)
    with open('validation_loss.pkl', 'wb') as f:
        pickle.dump(validation_loss_history, f)
    with open('validation_dsc.pkl', 'wb') as f:
        pickle.dump(validation_dsc_history, f)

    end = time.time()
    print(f'Done in {end - start}s')