from src.DCGAN import DCGAN
from src.image_loader import ImgLoader

img_loader = ImgLoader("E:\Datasets\keras_png_slices_data\keras_png_slices_train")
train_dataset = img_loader.load_to_tensor(0)

dcgan = DCGAN(64)
dcgan.train(train_dataset, 100)

