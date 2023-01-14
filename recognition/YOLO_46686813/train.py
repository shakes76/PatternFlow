from modules import yolo_loss
from modules import block_1, block_2, block_3, block_4, block_5, block_6, block_7
from modules import CustomLearningRateScheduler, lr_schedule, cp_callback
from dataset import bbox_to_list, savetxt_compact, load_data, img_path_list, tr_val_ts_split
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
import os

from tensorflow import keras


""" Loading the dataset """


im_path = "ISIC_tr"


X = img_path_list(im_path)

train_datasets = []

with open('target.txt', 'r') as f:
    train_datasets = train_datasets + f.readlines()

Y = []

for item in train_datasets:
  item = item.replace("\n", "").split(" ")
  arr = []
  for i in range(0, len(item)):
    arr.append(item[i])
  Y.append(arr)

print(Y[0])

x, y = load_data(X, Y)

x_train, x_test, x_val, y_train, y_test, y_val = tr_val_ts_split(x, y)


inputs = Input(shape=(448,448,3))
conv = block_1(inputs)
conv = block_2(conv)
conv = block_3(conv)
conv = block_4(conv)
conv = block_5(conv)
conv = block_6(conv)
output = block_7(conv)

model = Model(inputs, output)

model.compile(loss=yolo_loss, optimizer='adam')

model.fit(x_train, y_train, batch_size=16, epochs=30, validation_data=(x_val, y_val), callbacks=[CustomLearningRateScheduler(lr_schedule), cp_callback ])

model.save('saved_model/')