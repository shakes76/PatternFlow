"""
This is program for COMP3710 Report, resolving problem 6.
[Create a generative model of the OASIS brain or the OAI AKOA knee data set using a DCGAN that
has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.]
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load OASIS Dataset
DATASET_PATH = "../keras_png_slices_data/keras_png_slices_train/*"
filenames = glob.glob(DATASET_PATH)

if len(filenames) == 0:
    print("Error! Images not loaded!")
    exit()
else:
    n_datasize = len(filenames)

images = list()

for i in range(n_datasize):
    images.append(np.asarray(Image.open(filenames[i])))

images = np.array(images)

# Check dataset shape and verify the images
print(images.shape)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i])
plt.show()
