from train import model_train
from dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

Improved_unet = model_train()
# use trained model to predict the segmentation of given images
x_test = get_data("/Users/skywu/COMP3710-report/dataset/test_data/")
output = Improved_unet.predict(x_test)
# change the output size from 3D to 2D, some preparation for output the image
output = output[:, :, :, 1:2, :]
output = np.squeeze(output)
output = np.insert(output, 2, 0, axis=3)
output = np.where(output > 0.5, float(1), float(0))
# output the image
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(output[i])
plt.show()


# define dice loss to get the dice similarity coefficient
def dice_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


# compute the dice similarity coefficient
y_test = get_data("/Users/skywu/COMP3710-report/dataset/test_truth/", is_y=True)
y_test = y_test[:, :, :, 1:2, :]
y_test = np.squeeze(y_test)
y_test = np.insert(y_test, 2, 0, axis=3)
y_test = np.where(y_test > 0.5, float(1), float(0))
dice_similarity = 1-dice_loss(output, y_test)
print(dice_similarity.numpy())
