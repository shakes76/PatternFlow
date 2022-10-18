from PIL import Image
import os
from torch.utils.data import  DataLoader,  Dataset


class BrainDataset(Dataset):
    # There are 10400 AD images and 11120 NC images in the train folder. 
    # I further split the train images into train and validation sets, using approximatly 20% of the images for validation.
    # I manualy checked the junction datapoints where training and validation set splitting, to make sure not cover one patient's images in both sets.
    # Please note that  people from the same ID would only appear in either the train or validation set during training.


    def __init__(self, path, tfm=None, files=None, split='test'):
        super(BrainDataset).__init__()
        self.path = path
        AD_path = os.path.join(path, 'AD')
        NC_path = os.path.join(path, 'NC')
        # print(AD_path, 123)
        # print(os.path.join(AD_path,x) for x in os.listdir(AD_path) if x.endswith(".jpg",234))

        # Mark the label of AD as 1, NC as 0.
        self.AD_files = [(os.path.join(AD_path, x), 0)
                         for x in os.listdir(AD_path) if x.endswith(".jpeg")]
        self.NC_files = [(os.path.join(NC_path, x), 1)
                         for x in os.listdir(NC_path) if x.endswith(".jpeg")]

        if split == 'train':
            self.files = self.AD_files[:8320] + \
                self.NC_files[:8880]
        elif split == 'val':
            self.files = self.AD_files[8320:] + \
                self.NC_files[8880:]
        elif split == 'test':
            self.files = self.AD_files+self.NC_files
        elif split == 'pred':
            self.files = self.AD_files[:5]+self.NC_files[:5]
        if files != None:
            self.files = files
        # print(f"One {path} sample",self.files[0])
        # print(len(self.AD_files),len(self.NC_files),len(file))
        self.transform = tfm

    def __len__(self):
        return len(self.files)
    #Return the image data and label of the image
    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname[0])
        im = self.transform(im)
        label = fname[1]
        return im, label