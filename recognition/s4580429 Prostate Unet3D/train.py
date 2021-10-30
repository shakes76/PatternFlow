"""  imports  """
import tensorflow as tf
from tensorflow import keras as K
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

""" lazily convert paths to files into images / labels """
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

""" train/test/val split """
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

print("train", train_ds.cardinality())
print("test", test_ds.cardinality())
print("validation", val_ds.cardinality())

print("Successfully created, processed and split dataset!")

# dice coefficient function
def dice_coefficient(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    inters = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * inters) / (tf.reduce_sum(y_true + y_pred))

# the loss that we will use
total_loss = sm.losses.DiceLoss(class_weights=tf.ones((n_classes,))/n_classes) \
           + (1 * sm.losses.CategoricalFocalLoss())

# using metrics IOU and dice score to evaluate
metrics = [sm.metrics.IOUScore(threshold=0.5), dice_coefficient]

# create model
model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape=shape+(channels,), 
                encoder_weights=encoder_weights,
                activation=activation)

# compile model
model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

# fit model with train_ds, validate on val_ds
history = model.fit(train_ds, batch_size=batch_size, epochs=batch_num, verbose=1,
                  validation_data=val_ds)

model.save('/home/Student/s4580429/segment_out/3D_model_res10_full.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("/home/Student/s4580429/segment_out/loss.png")
plt.close()

acc = history.history['dice_coefficient']
val_acc = history.history['dice_coefficient']

plt.plot(epochs, acc, 'b', label='Training Dice')
plt.plot(epochs, val_acc, 'g', label='Validation Dice')
plt.title('Training and validation Dice Scores')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.savefig("/home/Student/s4580429/segment_out/dice.png")
plt.close()

#y_pred = model.predict(X_test)
#y_pred_argmax = tf.argmax(y_pred, axis=4)
#y_test_argmax = tf.argmax(y_test, axis=4)
#
#print(y_pred_argmax.shape)
#print(y_test_argmax.shape)
#print(tf.unique(tf.reshape(y_pred_argmax, (-1))))
#
#import random
#test_slice = random.randint(floor(128 * (1/3)), ceil(128 * (2/3)))
#test_img_ind = random.randint(0, len(X_test))
#test_img = X_test[test_img_ind]
#ground_truth = y_test[test_img_ind]
#
#test_img_input = tf.expand_dims(test_img, 0)
#test_img_input1 = preprocess_input(test_img_input)
#
#test_pred1 = model.predict(test_img_input1)
#test_prediction1 = tf.argmax(test_pred1, axis=4)[0,:,:,:]
#ground_truth_argmax = tf.argmax(ground_truth, axis=3)
#
#plt.figure(figsize=(12, 8))
#plt.subplot(231)
#plt.title('Testing Image')
#plt.imshow(test_img[test_slice,:,:,0], cmap='gray')
#plt.subplot(232)
#plt.title('Testing Label')
#plt.imshow(ground_truth_argmax[test_slice,:,:])
#plt.subplot(233)
#plt.title('Prediction on test image')
#plt.imshow(test_prediction1[test_slice,:,:])
#plt.savefig("/home/Student/s4580429/segment_out/slices.png")
#plt.close()