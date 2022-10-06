import argparse
from modules import Unet, Trainer
from torch.utils.data import DataLoader
from dataset import Dataset
from torch.optim import Adam

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('model', help="Path to model", type=str)
parser.add_argument('--num_images', '-t', help="Number of images to create", type=int)
parser.add_argument('--image_size', '-i', help="Dimensions of image to create. Needs to be same dimension the model was trained on. Default = 64", type=int)
parser.add_argument('--beta_schedule', '-s', help="Schdedule for calculating betas. Choose : linear, cosine, quadratic, sigmoid. Default: linear")
parser.add_argument('--timesteps', '-t', help="Number of timesteps in denoising pass. Default = 300", type=int)
parser.add_argument('--output', '-o', help="Output path to save images to", type=int)
parser.add_argument('--name', '-n', help="Name prefix for images", type=int)
args = parser.parse_args()

model = args.model

if args.num_images:
    num_images = args.num_images
else:
    num_images = 0
if args.image_size:
    image_size = args.image_size
else:
    image_size = 64
if args.output:
    output = args.output
else:
    output = './'
if args.name:
    name = args.name
else:
    name = "predict"
if args.beta_schedule:
    beta_schedule = args.beta_schedule
else:
    beta_schedule = 'linear'
if args.timesteps:
    timesteps = args.timesteps
else:
    timesteps = 300

#create model
model = Unet(image_size)
trainer = Trainer(model, img_size=image_size, timesteps=timesteps, start=0.0001, end=0.02)

#load model
trainer.load_model(model)

#create images
for i in range(num_images):
    trainer.generate_image(out + name + ".jpeg")