from imports import *
import os

""" A custom image loader dataset """
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


# Visualize our training data as a 3x3 grid of random samples
# imgs = ImageLoader("ADNI_AD_NC_2D//AD_NC//train//AD")
# imgs_dataloader = DataLoader(imgs, batch_size=32, shuffle=False)
# import matplotlib.pylab as plt
# cols, rows = 3, 3
# figure = plt.figure(figsize=(6, 6))
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(imgs), size=(1,)).item()
#     img = imgs[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()