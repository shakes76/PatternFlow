# File containing the source code for training, validating, testing and saving your model.
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
# Make sure to plot the losses and metrics during training

import modules
import dataset


# GOOGLE COLLAB TESTING
# !git clone https://github.com/ultralytics/yolov5  # clone
# %cd yolov5
# %pip install -qr requirements.txt  # install
#
# import torch
# from yolov5 import utils
# display = utils.notebook_init()

# %ls
# !unzip Archive.zip -d yolov5/data/

# %cd yolov5
# !python train.py --img 640 --batch 16 --epochs 5 --data ../dataset.yaml --weights yolov5s.pt

# !python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source data/images/training/ISIC_0000001.jpg

# from IPython.display import display
# from PIL import Image
#
# image_path = "runs/detect/exp5/ISIC_0000001.jpg"
# display(Image.open(image_path))