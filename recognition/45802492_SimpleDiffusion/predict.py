import argparse
from modules import Unet, Trainer
from torch.utils.data import DataLoader
from dataset import Dataset
from torch.optim import Adam

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('model', help="Path to model", type=str)
parser.add_argument('--output', '-o', help="Output path to save images to", type=str)
parser.add_argument('--num_images', '-i', help="Number of images to create", type=int)
parser.add_argument('--name', '-n', help="Name prefix for images", type=str)
args = parser.parse_args()

model = args.model

if args.num_images:
    num_images = args.num_images
else:
    num_images = 0
if args.output:
    output = args.output
else:
    output = './'
if args.name:
    name = args.name
else:
    name = "predict"

#Create trainer, pass in temp parameters
trainer = Trainer(img_size=64, timesteps=1000, start=0.0001, end=0.02)

#load model
trainer.load_model(model)

#create images
for i in range(num_images):
    trainer.generate_image(output + name + "{}.jpeg".format(i))