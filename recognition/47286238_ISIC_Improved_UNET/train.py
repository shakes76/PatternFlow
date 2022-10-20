import torch, torchvision
from dataset import Dataset
from modules import IUNET
import matplotlib.pyplot as plt
import time

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
    return 2 * torch.sum(intersection) / (torch.sum(a) + torch.sum(b))

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

if __name__ == '__main__':
    start = time.time()

    ## set device
    if not torch.cuda.is_available():
        print("CUDA not available for training! Aborting...")
    device = 'cuda'

    # TODO: optimal batch size?
    batch_size = 2
    dataset_train = Dataset(
        data_path='data/training/data', 
        truth_path='data/training/truth', 
        metadata_path='data/training/data/ISIC-2017_Training_Data_metadata.csv'
        )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    learning_rate = 1e-3
    num_epochs = 1

    model = IUNET(3, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # print('entering training loop')
    # for i in range(num_epochs):
    #     for batch_no, (img, mask) in enumerate(dataloader_train):
    #         img = img.to(device)
    #         mask = mask.to(device)

    #         out = model(img)
    #         loss = loss_fn(out[:,0,:,:], mask[:,0,:,:])

    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if batch_no == 500 / batch_size:
    #             break
    #     print("epoch done!")

    torch.save(model.state_dict(), 'model.pt')

    end = time.time()
    print(f'Done in {end - start}s')

    # with torch.no_grad():
    #     test = enumerate(dataloader_train)
    #     batch_no, (data, truth) = next(test)

    #     x = data[0]
    #     x = x.permute(1, 2, 0)
    #     print(x.shape)
    #     y = truth[0]
    #     y = y.permute(1,2,0)

    #     out = model(data.to('cuda'))
    #     # out = torchvision.transforms.Grayscale(3)(out)
    #     out_img = out.to('cpu')[0].permute(1,2,0)[:,:,0]
    #     # out_img2 = out.to('cpu')[0].permute(1,2,0)#[:,:,1]
    #     # out_img = torch.where(out_img > 0.5, 1.0, 0.0)

    #     print(out_img.max())
    #     print(out_img.min())

    #     print(y.max())
    #     print(y.min())

    #     fig = plt.figure()
    #     fig.add_subplot(3,1,1)
    #     plt.imshow(x)
    #     fig.add_subplot(3,1,2)
    #     plt.imshow(y)
    #     fig.add_subplot(3,1,3)
    #     plt.imshow(out_img)
    #     # fig.add_subplot(4,1,4)
    #     # plt.imshow(out_img2)
    #     plt.show()