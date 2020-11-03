import random
import tensorflow as tf
import imageio
import glob
import IPython


def get_ssim(gen_images, test_images):
    ssim_list = []
    for gim in gen_images:
        ssims = []
        for i, im in enumerate(test_images):
            if not random.randint(0, 2):
                # Compute SSIM over tf.float32 Tensors.
                ssims.append(tf.image.ssim(gim, im, max_val=1.0))
                # ssims.append(tf.image.ssim(gim, im, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))
                ssim_list.append(max(ssims))
    return ssim_list


def generate_gif(path):
    anim_file = 'dcgan_training.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(path + 'image*.jpg')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = i
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    IPython.display.Image(filename=anim_file)
