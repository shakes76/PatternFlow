import random
import tensorflow as tf


def get_ssim(gen_images, test_images):
    ssim_list = []
    for gim in gen_images:
        ssims = []
        for i, im in enumerate(test_images):
            if not random.randint(0, 2):
                ssims.append(tf.image.ssim(gim, im, max_val=255))
        ssim_list.append(sum(ssims) / len(ssims))
    return ssim_list
