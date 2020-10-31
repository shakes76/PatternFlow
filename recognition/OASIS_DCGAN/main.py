import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import recognition.OASIS_DCGAN.dcgan as dcgan
import recognition.OASIS_DCGAN.training as training
from recognition.OASIS_DCGAN.dataset import read_dataset
from recognition.OASIS_DCGAN.training import train
from recognition.OASIS_DCGAN.config import Config

# Model summarization ####################################################
# training.generator.summary()
# training.discriminator.summary()

# Training process #######################################################
if Config.Train:
    train_dataset = read_dataset(switch='train')
    train(train_dataset, Config.EPOCHS)

# SSIM measure ###########################################################
gen_images = training.generator(training.seed, training=False)  # restored checkpoint needed
# # show generatedd figure
# plt.figure(figsize=(4, 4))
# for i in range(len(gen_images)):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(((gen_images[i] + 1.0) / 2.0 * 255), cmap='gray')
#     plt.axis('off')
# plt.show()
test_images = read_dataset(switch='test')
sum_ssim = []
print('Measuring The Structural Similarity:')
for gim in gen_images:
    ssim_list = []
    for i, im in enumerate(test_images):
        if not random.randint(0, 2):
            ssim_list.append(tf.image.ssim(gim, im, max_val=255))
    sum_ssim.append(sum(ssim_list) / len(ssim_list))
print('SSIM = %.3f%%' % (100 * sum(sum_ssim) / len(sum_ssim)))

