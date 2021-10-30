import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot


def load_image(file):
    # loads the jpeg images into usable images for the model
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0  # need to divide
    return img


def load_seg(file):
    # loads the png segmentation images into usable images for the model
    seg = tf.io.read_file(file)
    seg = tf.image.decode_png(seg, channels=1)  # gray scale
    seg = tf.image.resize(seg, (256, 256))
    seg = tf.cast(seg, tf.float32) / 255.0  # need to divide
    seg = tf.math.round(seg)  # round so that it is binary
    return seg


def process_images(image_path, mask_path):
    # load both the images and segmentation images
    image = load_image(image_path)
    segmentation = load_seg(mask_path)

    # reshape them both to be 256, 256, 1
    image = tf.reshape(image, (256, 256, 1))
    segmentation = tf.reshape(segmentation, (256, 256, 1))
    return image, segmentation


def prepare_images(max_images):
    # get the path of the images
    x_images = 'C:\\Users\\Mark\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1-2_Training_Input_x2\\*.jpg'
    mask_images = 'C:\\Users\\Mark\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1_Training_GroundTruth_x2\\*.png'
    x_images = sorted(glob.glob(x_images))
    mask_images = sorted(glob.glob(mask_images))
    # take only a certain number of images to be used
    new_x = x_images[0:max_images]
    new_mask = mask_images[0:max_images]

    print("Length of total images split 60% in training, 20% testing and 20% validation:")
    print(len(new_x))
    print(len(new_mask))
    # split the images into training, testing and validation
    # 60% training, 20% testing, 20% val
    X_train, X_test, y_train, y_test = train_test_split(new_x, new_mask, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    # Put the images into a dataset from a list
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # shuffle the images
    final_train = train_ds.shuffle(len(X_train))
    final_test = test_ds.shuffle(len(X_test))
    final_val = val_ds.shuffle(len(X_val))

    # put the images into a map
    final_train = final_train.map(process_images)
    final_test = final_test.map(process_images)
    final_val = final_val.map(process_images)

    return final_train, final_test, final_val

