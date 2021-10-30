"""  imports  """
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras as K
from tensorflow._api.v2 import data
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.python.data.ops.dataset_ops import TakeDataset
device = tf.config.experimental.list_physical_devices("GPU")
if device == '[]':
    raise SystemError("No GPU!!")
print("Found GPU at: {}".format(device))

import segmentation_models_3D as sm
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from math import floor, ceil
import glob
import re

""" model parameters """
shape = nib.load("/home/Student/s4580429/dataset/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz").get_fdata().shape
channels = 3
n_classes = 6
BACKBONE = 'resnet50'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
encoder_weights = 'imagenet'
activation = 'softmax'

LR = 0.0001
batch_size = 1
batch_num = 10
optim = K.optimizers.Adam(LR)

def getScans(dataFolder, labelsFolder):
    dataFiles = sorted(glob.glob(dataFolder + "*"))
    labelsFiles = sorted(glob.glob(labelsFolder + "*"))
    
    return list(zip(dataFiles, labelsFiles))

fileNames = getScans("/home/Student/s4580429/dataset/semantic_MRs_anon/", "/home/Student/s4580429/dataset/semantic_labels_anon/")
total_images = len(fileNames)

""" get each group of cases' positions """
ind_of_first_case = []
i = 0
lastCase = None
for path, _ in fileNames:
    x = re.search("Case_(\d+)_", path).group(1)
    if lastCase != x:
        ind_of_first_case.append(i)
        lastCase = x
    i += 1

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(range(len(fileNames))), tf.constant(range(len(fileNames)))))

preprocess_input = sm.get_preprocessing(BACKBONE)

def im_file_to_tensor(dataPath, labelPath):
    def _im_file_to_tensor(dataPath, labelPath):
        dp, lp = fileNames[dataPath]
        image = tf.convert_to_tensor(nib.load(dp).get_fdata().astype('float32'))
        image = tf.cast(image, tf.float32)
        image = tf.stack((image,)*channels, axis=-1)
        image = preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        labels = tf.convert_to_tensor(nib.load(lp).get_fdata().astype('float32'))
        labels = to_categorical(tf.expand_dims(labels, axis=3), num_classes=n_classes)
        labels = tf.cast(labels, tf.float32)
        labels = tf.expand_dims(labels, axis=0)
        return image, labels
    data, label = tf.py_function(_im_file_to_tensor, 
                                 inp=(dataPath, labelPath), 
                                 Tout=(tf.float32, tf.float32))
    data.set_shape((1,) + shape + (channels,))
    label.set_shape((1,) + shape + (n_classes,))
    return data, label

dataset = dataset.map(im_file_to_tensor,
    num_parallel_calls=tf.data.AUTOTUNE)

# TODO try to add shuffle
i_train = floor(0.7 * total_images)
while i_train not in ind_of_first_case:
    i_train += 1
rem = total_images - i_train - 1
i_test = floor(i_train + rem/2)
while i_test not in ind_of_first_case:
    i_test += 1
i_test -= i_train

train_ds = dataset.take(i_train)
test_ds = dataset.skip(i_train).take(i_test)
val_ds = dataset.skip(i_train + i_test)

print("Successfully created, processed and split dataset!")

dice_loss = sm.losses.DiceLoss(class_weights=tf.ones((n_classes,))/n_classes) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model = load_model('/home/Student/s4580429/segment_out/3D_model_vgg16_5epochs_full3d_parr.h5', compile=False)

#y_pred = model.predict(test_ds)

import random
test_img_number = random.randint(0, len(test_ds))
test_img = None
ground_truth = None
for image, label in test_ds.take(1):
    test_img = image
    ground_truth = label

test_pred1 = model.predict(test_img)
test_prediction1 = tf.argmax(test_pred1, axis=4)[0,:,:,:]
ground_truth_argmax = tf.argmax(ground_truth, axis=4)
print("test_img:", test_img)
print("test_pred:", test_prediction1)
print("ground_truth:", ground_truth_argmax)

aff = nib.load("/home/Student/s4580429/dataset/semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz").affine
nib.save(nib.Nifti1Image(test_prediction1.numpy(), aff), "/home/Student/s4580429/segment_out/labels_pred.nii")

import random
test_slice = random.randint(floor(128 * (1/3)), ceil(128 * (2/3)))

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[0, :,:,test_slice,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth_argmax[0,:,:,test_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1[:,:,test_slice])
plt.savefig("/home/Student/s4580429/segment_out/compar.png")