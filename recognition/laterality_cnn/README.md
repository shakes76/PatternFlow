# OAI AKOA Dataset Laterality Classification

*description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.*
*description and explanation of the working principles of the algorithm implemented and the problem it solves*


## Pre-requisites
* Python 3.5-3.8
* Tensorflow 2.1.0
  * If using Python 3.8, Tensorflow 2.2 or later is required.
* Matplotlib 3.3.2

## Usage
### cnn.py
The Convolutional Neural Network module can be used in external scripts by importing the class `CNNModel` from *cnn.py* and using 

```python
model = CNNModel(num_classes=<num_classes>)
```
where *<num_classes>* is the number of classes your data contains.

### classify_laterality.py
The driver script *classify_laterality.py* can be run from the cmd line using

```
python classify_laterality.py <path_to_data_folder>
```

where *<path_to_data_folder>* is the path to the OAI AKOA data folder e.g. *"C:\Users\\<user\>\\.keras\datasets\AKOA_Analysis"*.

## Examples
An example usage of the driver script on the OAI AKOA dataset produces the following
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv_block (ConvBlock)       multiple                  9568
_________________________________________________________________
conv_block_1 (ConvBlock)     multiple                  55424
_________________________________________________________________
conv_block_2 (ConvBlock)     multiple                  221440
_________________________________________________________________
flatten (Flatten)            multiple                  0
_________________________________________________________________
dense (Dense)                multiple                  11878528
_________________________________________________________________
dropout_3 (Dropout)          multiple                  0
_________________________________________________________________
dense_1 (Dense)              multiple                  258
=================================================================
Total params: 12,165,218
Trainable params: 12,165,218
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
2020-11-02 16:22:30.660337: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:172] Filling up shuffle buffer (this may take a while): 5436 of 12360
2020-11-02 16:22:40.660439: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:172] Filling up shuffle buffer (this may take a while): 10996 of 12360
2020-11-02 16:22:43.193683: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:221] Shuffle buffer filled.
387/387 [==============================] - ETA: 0s - loss: 0.1325 - accuracy: 0.9390 
Epoch 00001: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.1325 - accuracy: 0.9390 - val_loss: 0.1581 - val_accuracy: 0.9723
Epoch 2/10
386/387 [============================>.] - ETA: 0s - loss: 0.0100 - accuracy: 0.9972     
Epoch 00002: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 90ms/step - loss: 0.0100 - accuracy: 0.9972 - val_loss: 0.1582 - val_accuracy: 0.9757
Epoch 3/10
386/387 [============================>.] - ETA: 0s - loss: 0.0156 - accuracy: 0.9958 
Epoch 00003: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 93ms/step - loss: 0.0156 - accuracy: 0.9958 - val_loss: 0.1379 - val_accuracy: 0.9794
Epoch 4/10
386/387 [============================>.] - ETA: 0s - loss: 0.0011 - accuracy: 0.9997     
Epoch 00004: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.2879 - val_accuracy: 0.9679
Epoch 5/10
386/387 [============================>.] - ETA: 0s - loss: 0.0041 - accuracy: 0.9988     
Epoch 00005: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.0041 - accuracy: 0.9988 - val_loss: 0.3428 - val_accuracy: 0.9706
Epoch 6/10
386/387 [============================>.] - ETA: 0s - loss: 0.0026 - accuracy: 0.9991     
Epoch 00006: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.0026 - accuracy: 0.9991 - val_loss: 0.6782 - val_accuracy: 0.9726
Epoch 7/10
386/387 [============================>.] - ETA: 0s - loss: 0.0170 - accuracy: 0.9961
386/387 [============================>.] - ETA: 0s - loss: 5.3358e-04 - accuracy: 0.9998
Epoch 00008: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 93ms/step - loss: 5.3323e-04 - accuracy: 0.9998 - val_loss: 0.1791 - val_accuracy: 0.9784
Epoch 9/10
386/387 [============================>.] - ETA: 0s - loss: 0.0021 - accuracy: 0.9997
Epoch 00009: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 91ms/step - loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.1515 - val_accuracy: 0.9801
Epoch 10/10
386/387 [============================>.] - ETA: 0s - loss: 0.0022 - accuracy: 0.9994
Epoch 00010: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 90ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.1100 - val_accuracy: 0.9861
Training set:
387/387 - 9s - loss: 1.1706e-06 - accuracy: 1.0000
Validation set:
93/93 - 2s - loss: 0.1100 - accuracy: 0.9861
Test set:
105/105 - 2s - loss: 0.0030 - accuracy: 0.9991
```

![Accuracy vs Epochs](/plots/accuracy.png)

![Loss vs Epochs](/plots/loss.png)

## Dataset Splitting


## Authors
Khang Nguyen
