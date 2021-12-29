import tensorflow as tf
import os
import improvedUNetModel

IMG_SIZE = (256, 256)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
DATA_DIRECTORY = __location__ + "\\data"
IMG_FOLDER = DATA_DIRECTORY + "\\image"
MASK_FOLDER = DATA_DIRECTORY + "\\mask"

# The images and masks are different file types so I had to pull some shenanigans to get them to play nice with tf
# Datasets. I tried like 5 different ways using other tf functions so it's like this now *shrug*
images_list = tf.io.gfile.glob(str(IMG_FOLDER + '/*.jpg'))
masks_list = tf.io.gfile.glob(str(MASK_FOLDER + '/*.png'))


def normalize(image_path):
    """
    Reads and normalizes the input images

    Parameters
    ----------
    image_path
    the path to the image.

    Returns
    -------
    float32 : the normalized image
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def decode_img(image_path, mask_path):
    image = normalize(image_path)
    mask = normalize(mask_path)
    return image, mask


def data_combination(images, masks):
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    # I have to .shuffle before .map here as my computer crashes if I try to shuffle the tensors after
    # they've been loaded
    data = data.shuffle(len(images))
    data = data.map(decode_img)

    return data


image_data = data_combination(images_list, masks_list)
test_dataset = image_data.take(1000)
remaining = image_data.skip(1000)
val_dataset = remaining.take(500)
train_dataset = remaining.skip(500)

model = improvedUNetModel.improvedUNet()
model.fit(train_dataset.batch(16), epochs=8, validation_data=val_dataset.batch(16))

model.evaluate(test_dataset.bacth(16))
