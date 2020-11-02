import matplotlib.pyplot as plt
import sys

import eval
import training
import dataset
from config import Config


def main(arg_list):
    TRAIN = True if arg_list[0] == 'train' else False
    RESTORE = True if arg_list[1] == 'restore' else False
    # Model summarization
    # training.generator.summary()
    # training.discriminator.summary()

    # Training process
    if TRAIN:
        train_dataset = dataset.read_dataset(switch='train')
        training.train(train_dataset, Config.EPOCHS)
    if RESTORE:
        training.restore_checkpoint()

    # SSIM measure
    gen_images = training.generator(training.SEED, training=False)  # restored checkpoint needed
    # show generated 16 figuresS
    plt.ion()
    plt.figure(figsize=(4, 4))
    for i in range(len(gen_images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(((gen_images[i] + 1.0) / 2.0 * 255), cmap='gray')
        plt.axis('off')
    plt.pause(2)

    test_images = dataset.read_dataset(switch='test')
    print('Measuring The Structural Similarity...')
    ssim_list = eval.get_ssim(gen_images, test_images)
    # ssim = sum(ssim_list) / len(ssim_list) # average mean_ssim for 16 gen_fig
    ssim = max(ssim_list)  # maximum mean_ssim for 16 gen_fig
    print('SSIM = %.3f%%' % (100 * ssim))


if __name__ == '__main__':
    main(sys.argv[1:])
