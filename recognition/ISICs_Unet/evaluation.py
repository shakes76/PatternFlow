"""
File name: evaluation.py
Author: Thomas Chen
Date created: 11/3/2020
Date last modified: 11/24/2020
Python Version: 3
"""
import numpy as np
import matplotlib.pyplot as plt

from data import test_generator, test_data
from setting import *
from unet import UNet

"""
Do evaluation with result from train.py
"""
model = UNet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.load_weights("unet.hdf5")
results = model.evaluate(test_generator,
                         verbose=1,
                         steps=len(test_data) / BATCH_SIZE,
                         return_dict=True)
print('Evaluation dice coefficient is', results['dice_coef'])

"""
plot image for samples
"""
batches = 2
plt.figure(figsize=(10, 10))
for i in range(batches):
    img, mask = next(test_generator)
    seg = (model.predict(img) > 0.5).astype(np.float32)
    for b in range(BATCH_SIZE):
        plt.subplot(batches * BATCH_SIZE, 3, b * 3 + 1 + i * BATCH_SIZE * 3)
        plt.imshow(img[b])
        if i + b == 0:
            plt.title('input image')
        plt.subplot(batches * BATCH_SIZE, 3, b * 3 + 2 + i * BATCH_SIZE * 3)
        plt.imshow(seg[b])
        if i + b == 0:
            plt.title('predicted mask')
        plt.subplot(batches * BATCH_SIZE, 3, b * 3 + 3 + i * BATCH_SIZE * 3)
        plt.imshow(mask[b])
        if i + b == 0:
            plt.title('ground truth')
plt.savefig('visualization of some predictions')
plt.show()

