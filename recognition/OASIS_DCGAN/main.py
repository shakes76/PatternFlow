import matplotlib.pyplot as plt

import training
import eval
from dataset import read_dataset
from training import train
from config import Config

# Model summarization ####################################################
# training.generator.summary()
# training.discriminator.summary()

# Training process #######################################################
if Config.Train:
    train_dataset = read_dataset(switch='train')
    train(train_dataset, Config.EPOCHS)

# SSIM measure ###########################################################
gen_images = training.generator(training.seed, training=False)  # restored checkpoint needed
# show generated 16 figuresS
plt.ion()
plt.figure(figsize=(4, 4))
for i in range(len(gen_images)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(((gen_images[i] + 1.0) / 2.0 * 255), cmap='gray')
    plt.axis('off')
plt.pause(0.1)

test_images = read_dataset(switch='test')
print('Measuring The Structural Similarity...')
ssim_list = eval.get_ssim(gen_images, test_images)
print(ssim_list)
# ssim = sum(ssim_list) / len(ssim_list) # average mean_ssim for 16 gen_fig
ssim = max(ssim_list)  # maximum mean_ssim for 16 gen_fig
print('SSIM = %.3f%%' % ssim * 100)
