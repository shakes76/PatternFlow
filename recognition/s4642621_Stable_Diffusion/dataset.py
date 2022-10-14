from imports import *
import os

"""
A custom image loader dataset
"""
class ImageLoader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.total_imgs = os.listdir(img_dir)
        self.total_imgs.sort()

        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        transform = torchvision.transforms.ToTensor()
        image = os.path.join(self.img_dir, self.total_imgs[index])
        image = transform(Image.open(image).convert("L"))
        if self.transform:
            image = self.transform(image)
        return image

"""
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_dataset(batch_size=8, image_resize=256, val_set=False, ad_train_path="ADNI_DATA/AD_NC/train/AD", nc_train_path="ADNI_DATA/AD_NC/train/NC", ad_test_path="ADNI_DATA/AD_NC/test/AD", nc_test_path="ADNI_DATA/AD_NC/test/NC"):
    # transform image from [1,255] to [0,1], and scale linearly into [-1,1]
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                    Lambda(lambda t: (t * 2) - 1),  # scale linearly into [-1,1]
                ])

    # if we are loading in a validation set
    # take 10% of both the ad and nd datasets
    # i.e. add every 10th image to the dataset
    if val_set:
        ad_val = os.listdir(ad_test_path)
        ad_val.sort()
        nc_val = os.listdir(nc_test_path)
        nc_val.sort()

        val = []
        # take every 10th image from both datasets
        # i.e. 10% of the total dataset
        for index, image in enumerate(ad_val):
            if index % 10 == 0:
                val.append(image)
        for index, image in enumerate(nc_val):
            if index % 10 == 0:
                val.append(image)

        # load validate data, with the above transform applied
        val_imgs = ImageLoader(val, transform=transform)
        return DataLoader(val_imgs, batch_size=batch_size, shuffle=False, num_workers=1)
    else:
        # load data, with the above transform applied
        train_ad_imgs = ImageLoader(ad_train_path,
                                    transform=transform)
        train_nc_imgs = ImageLoader(nc_train_path,
                                    transform=transform)
        # combine ad and nc train datasets, to increase total number of images for training
        total_imgs = torch.utils.data.ConcatDataset([train_ad_imgs, train_nc_imgs])
        return DataLoader(total_imgs, batch_size=batch_size, shuffle=False, num_workers=1)
