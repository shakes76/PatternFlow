import argparse
from modules import Unet, Trainer
from torch.utils.data import DataLoader
from dataset import Dataset
from torch.optim import Adam

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('name', help="Model name, call me something cool", type=str)
parser.add_argument('path', help="Path to dataset", type=str)
parser.add_argument('--timesteps', '-t', help="Number of timesteps in denoising pass. Default = 300", type=int)
parser.add_argument('--epochs', '-e', help="Number of epochs to train for. Default = 100", type=int)
parser.add_argument('--batch_size', '-b', help="Training mini-batch size. Default = 128", type=int)
parser.add_argument('--image_size', '-i', help="Dimensions to resize all images to. eg 255 -> 255x255 Default = 64", type=int)
parser.add_argument('--beta_schedule', '-s', help="Schdedule for calculating betas. Choose : linear, cosine, quadratic, sigmoid. Default: linear")
parser.add_argument('--disable_images', help="Images will not be output during training", action='store_false')
parser.add_argument('--disable_tensorboard', help="Tensorboard support will be disabled", action='store_false')
args = parser.parse_args()

#Load values from arguments
name = args.name
path = args.path

if args.timesteps:
    timesteps = args.timesteps
else:
    timesteps = 300

if args.epochs:
    epochs = args.epochs
else:
    epochs = 100

if args.batch_size:
    batch_size = args.batch_size
else:
    batch_size = 128

if args.image_size:
    image_size = args.image_size
else:
    image_size = 64

if args.disable_images:
    disable_images = True
else:
    disable_images = False

if args.disable_tensorboard:
    disable_tensorboard = True
else:
    disable_tensorboard = False
if args.beta_schedule:
    beta_schedule = args.beta_schedule
else:
    beta_schedule = 'linear'

#load dataset
ds = Dataset(path, img_size=image_size)
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

#create model
model = Unet(image_size)

#Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)

#create trainer
trainer = Trainer(model, img_size=image_size, timesteps=timesteps, start=0.0001, end=0.02, create_images= disable_images, tensorboard = disable_tensorboard, schedule=beta_schedule)

#train model
trainer.fit(dataloader, epochs, optimizer)

#Save model
trainer.save_model(name + ".pth")