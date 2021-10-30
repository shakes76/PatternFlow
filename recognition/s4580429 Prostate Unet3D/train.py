print("Hello world! We've changed!")

import tensorflow as tf
from tensorflow import keras as K
device = tf.config.experimental.list_physical_devices("GPU")
if device == '[]':
    raise SystemError("No GPU!!")
print("Found GPU at: {}".format(device))

import segmentation_models_3D as sm
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from math import floor, ceil

image = nib.load("/home/Student/s4580429/dataset/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz")
image_patch = patchify(image.get_fdata(), (64,64,64), step=64)

label = nib.load("/home/Student/s4580429/dataset/semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz")
label_patch = patchify(label.get_fdata(), (64,64,64), step=64)

input_images = tf.reshape(image_patch, (-1, 64, 64, 64))
input_labels = tf.reshape(label_patch, (-1, 64, 64, 64))

n_classes = 6

if len(input_images) - floor(0.8 * len(input_images)) % 2 == 0:
    train_size = floor(0.8 * len(input_images))
else:
    train_size = ceil(0.8 * len(input_images))

val_size = test_size = (len(input_images) - train_size) // 2
ds = tf.data.Dataset.from_tensor_slices((input_images, input_labels))
ds = ds.shuffle(10000)
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)
test_ds = ds.skip(train_size + val_size)

dataset_to_tensor = lambda tf_dataset : tf.convert_to_tensor([x for x in tf_dataset])
process = lambda x, fun : tf.stack((x[:, 0, :, :, :],)*3, axis=-1) if fun == 0 else \
        to_categorical(tf.expand_dims(x[:, 1, :, :, :], axis=4), num_classes=n_classes)

X_train = dataset_to_tensor(train_ds)
y_train = process(X_train, 1)
X_train = process(X_train, 0)
X_test = dataset_to_tensor(test_ds)
y_test = process(X_test, 1)
X_test = process(X_test, 0)
X_val = dataset_to_tensor(val_ds)
y_val = process(X_val, 1)
X_val = process(X_val, 0)

print("train", X_train.shape, y_train.shape, type(X_train), type(y_train))
print("test", X_test.shape, y_test.shape, type(X_test), type(y_test))
print("validation", X_val.shape, y_val.shape, type(X_val), type(y_val))

print("Successfully loaded and converted data!")

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    inters = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * inters) / (tf.reduce_sum(y_true + y_pred))

encoder_weights = 'imagenet'
BACKBONE = 'resnet50'
activation = 'softmax'
patch_size = 64
channels = 3

LR = 0.0001
batch_size = 1
batch_num = 50
optim = K.optimizers.Adam(LR)

total_loss = sm.losses.DiceLoss() + (1 * sm.losses.CategoricalFocalLoss())

metrics = [sm.metrics.IOUScore(threshold=0.5), dice_coefficient]

preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prep = preprocess_input(X_train)
X_val_prep = preprocess_input(X_val)

model = sm.Unet(BACKBONE, classes=n_classes,
                input_shape=(patch_size, patch_size, patch_size, channels),
                encoder_weights=encoder_weights,
                activation=activation)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

history = model.fit(X_train_prep, y_train, batch_size=batch_size,
                  epochs=batch_num, verbose=1,
                  validation_data=(X_val_prep, y_test))

model.save('/home/Student/s4580429/segment_out/3D_model_res50_patches.h5')

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
plt.ylabel('IOU')
plt.legend()
plt.savefig("/home/Student/s4580429/segment_out/dice.png")
plt.close()

y_pred = model.predict(X_test)
y_pred_argmax = tf.argmax(y_pred, axis=4)
y_test_argmax = tf.argmax(y_test, axis=4)

print(y_pred_argmax.shape)
print(y_test_argmax.shape)
print(tf.unique(tf.reshape(y_pred_argmax, (-1))))