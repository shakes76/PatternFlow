import tensorflow as tf
import nibabel


# Labels:
# Background = 0
# Body = 1
# Bones = 2
# Bladder = 3
# Rectum = 4
# Prostate = 5


def get_nifti_data(file_name):
    # tf.string(file_name)
    # bits = tf.io.read_file(file_name)
    img = nibabel.load(file_name).get_fdata()
    return img


def one_hot(file_name):
    mask = get_nifti_data(file_name)
    bg = mask == 0
    bg = tf.where(bg == True, 1, 0)
    body = mask == 1
    body = tf.where(body == True, 1, 0)
    bones = mask == 2
    bones = tf.where(bones == True, 1, 0)
    bladder = mask == 3
    bladder = tf.where(bladder == True, 1, 0)
    rectum = mask == 4
    rectum = tf.where(rectum == True, 1, 0)
    prostate = mask == 5
    prostate = tf.where(prostate == True, 1, 0)
    return tf.concat((bg, body, bones, bladder, rectum, prostate), axis=-1)


def map_fn(image, mask):
    image = get_nifti_data(image)
    mask = one_hot(mask)
    return image, mask
