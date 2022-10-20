# %%
from dataset import *
from modules import *
from train import *
from predict import *

# %%
# get required data
train_x = load_images('C:/TechnoCore/2022/COMP3710/project/train_x')
train_y = load_labels('C:/TechnoCore/2022/COMP3710/project/train_y')
valid_x = load_images('C:/TechnoCore/2022/COMP3710/project/valid_x')
valid_y = load_labels('C:/TechnoCore/2022/COMP3710/project/valid_y')

# %%
unet, results = train_unet(train_x, train_y, valid_x, valid_y)

# %%
