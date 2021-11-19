# Using YOLOv5 to Identify Lesions in the ISIC Dataset

ISIC is a dataset...

## YOLOv5 Overview


## Usage


## Data Processing


## Training Methodology
My training methodology varied and evolved many times over the course of the project. At first I was just going to pass and images that had to mention 1024 x 1024, but I realised that six 640 x 640 was more appropriate. This is because YOLOv5 was designed for that image dimension. I also attempted to augment the data using a few different tricks. Additionally, I was careful not to also augment the test or validation sept data as this is completely inappropriate.

## Results
