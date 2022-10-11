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
parser.add_argument('--seed', '-s', help="Random seed. Passing the same random seed yields deterministic images", type=int)
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
if args.seed:
    seed = args.seed
else:
    seed = None

#create model
model = Unet(64)

#Create trainer, pass in temp parameters
trainer = Trainer(model, img_size=64, timesteps=300, start=0.0001, end=0.02)

#load model
trainer.load_model(model)

#create images
for i in range(num_images):
    trainer.generate_image(out + name + "{}.jpeg".format(i), seed=seed)