# OAI AKOA Dataset Laterality Classification
In order to laterally classify the OAI AKOA knee data set, a Convolution Neural Network (CNN) based classifier was modelled using [Tensorflow](https://www.tensorflow.org/). 
The dataset consists of 18,680 MRIs (greyscale) of knees, labelled by patient and whether they are of the left or right knee. Approximately 58% of the MRIs are '*right'* classified with the rest being *'left'* classified. The dataset is split into 3 subsets - training, validation and testing, as detailed in the **Dataset Splitting** section.

The CNN model strings together multiple layers and then ties them to fully connected (dense) layers to produce classified outputs. The configuration of the CNN is as follows:

```python
(input: 228 x 260 dimenson images (greyscale))
  ↓
2 x [Convolution (32 filters, (3 x 3) kernel size, relu activation)]
  ↓
[Max Pooling ((2 x 2) pool size)]
  ↓
[Dropout (0.2 rate)]
  ↓
2 x [Convolution (64 filters, (3 x 3) kernel size, relu activation)]
  ↓
[Max Pooling ((2 x 2) pool size)]
  ↓
[Dropout (0.2 rate)]
  ↓
2 x [Convolution (128 filters, (3 x 3) kernel size, relu activation)]
  ↓
[Max Pooling ((2 x 2) pool size)]
  ↓
[Dropout (0.2 rate)]
  ↓
[Flatten ()]
  ↓
[Dense (128 units, relu activation)]
  ↓
[Dropout (0.5 rate)]
  ↓
[Dense (<num_classes> units, softmax activation)]
  ↓
(output: logits of probability distribution over <num_classses> classes)
```

The above CNN configuration is capable of achieving 99% accuracy on a test set within 10 epochs as shown in the **Example** section.

## Pre-requisites
* Python 3.5-3.8
* Tensorflow 2.1.0
  * If using Python 3.8, Tensorflow 2.2 or later is required.
* Matplotlib 3.3.2 (for plots)

## Usage
### cnn.py
The CNN module can be used in external scripts by importing the class `CNNModel` from *cnn.py* and using 

```python
model = CNNModel(num_classes=<num_classes>)
```
where *<num_classes>* is the number of classes your data contains.

### classify_laterality.py
The driver script *classify_laterality.py* can be run from the cmd line using

```
>python classify_laterality.py <path_to_data_folder>
```

where *<path_to_data_folder>* is the path to the OAI AKOA data folder e.g. *"C:\Users\\<user\>\\.keras\datasets\AKOA_Analysis"*.

Hyperparameters that can be adjusted (*Batch size, learning rate, number of epochs*) in `classify_laterality.py` are at the top of the script under `#Parameters`. Additionally, the script saves the weights of the model via checkpoints after every epoch. The location of the checkpoint file can be changed via the variable `checkpoint_path` (ln 161). By default, the checkpoint is saved to `training/ckpt01.ckpt`. Loading of the weights can then be done by uncommenting the line `#model.load_weights(checkpoint_path).expect_partial()` (ln 191). If you wish to simply load the weights without doing anymore fitting, the `model.fit(...)` (ln 194) line should be commented out, as well as all the code for plotting (lns 204 - 223).

## Example
An example usage of the driver script using `>python classify_laterality.py <path_to_data_folder>` on the OAI AKOA dataset produces the following
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
2020-11-02 16:51:19.488335: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:172] Filling up shuffle buffer (this may take a while): 5606 of 12360
2020-11-02 16:51:29.488152: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:172] Filling up shuffle buffer (this may take a while): 11134 of 12360
2020-11-02 16:51:31.735819: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:221] Shuffle buffer filled.
387/387 [==============================] - ETA: 0s - loss: 0.1058 - accuracy: 0.9518 
Epoch 00001: saving model to training\ckpt01.ckpt
387/387 [==============================] - 37s 95ms/step - loss: 0.1058 - accuracy: 0.9518 - val_loss: 0.1579 - val_accuracy: 0.9693
Epoch 2/10
386/387 [============================>.] - ETA: 0s - loss: 0.0053 - accuracy: 0.9982 
Epoch 00002: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 90ms/step - loss: 0.0053 - accuracy: 0.9982 - val_loss: 0.3417 - val_accuracy: 0.9679
Epoch 3/10
386/387 [============================>.] - ETA: 0s - loss: 0.0076 - accuracy: 0.9972     
Epoch 00003: saving model to training\ckpt01.ckpt
387/387 [==============================] - 34s 89ms/step - loss: 0.0076 - accuracy: 0.9972 - val_loss: 0.3153 - val_accuracy: 0.9662
Epoch 4/10
386/387 [============================>.] - ETA: 0s - loss: 0.0189 - accuracy: 0.9952     
Epoch 00004: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 91ms/step - loss: 0.0189 - accuracy: 0.9952 - val_loss: 0.2524 - val_accuracy: 0.9639
Epoch 5/10
386/387 [============================>.] - ETA: 0s - loss: 0.0057 - accuracy: 0.9982     
Epoch 00005: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.0057 - accuracy: 0.9982 - val_loss: 0.2695 - val_accuracy: 0.9655
Epoch 6/10
386/387 [============================>.] - ETA: 0s - loss: 1.8707e-04 - accuracy: 1.0000 
Epoch 00006: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 91ms/step - loss: 1.8695e-04 - accuracy: 1.0000 - val_loss: 0.3639 - val_accuracy: 0.9652
Epoch 7/10
386/387 [============================>.] - ETA: 0s - loss: 0.0022 - accuracy: 0.9997
Epoch 00007: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 93ms/step - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.3479 - val_accuracy: 0.9649
Epoch 8/10
386/387 [============================>.] - ETA: 0s - loss: 6.8052e-04 - accuracy: 0.9996
Epoch 00008: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 91ms/step - loss: 6.8008e-04 - accuracy: 0.9996 - val_loss: 0.1887 - val_accuracy: 0.9787
Epoch 9/10
386/387 [============================>.] - ETA: 0s - loss: 0.0153 - accuracy: 0.9969
Epoch 00009: saving model to training\ckpt01.ckpt
387/387 [==============================] - 35s 90ms/step - loss: 0.0154 - accuracy: 0.9968 - val_loss: 0.4792 - val_accuracy: 0.9618
Epoch 10/10
386/387 [============================>.] - ETA: 0s - loss: 0.0036 - accuracy: 0.9989
Epoch 00010: saving model to training\ckpt01.ckpt
387/387 [==============================] - 36s 92ms/step - loss: 0.0036 - accuracy: 0.9989 - val_loss: 0.4388 - val_accuracy: 0.9679
Training set:
387/387 - 9s - loss: 7.6011e-07 - accuracy: 1.0000
Validation set:
93/93 - 2s - loss: 0.4388 - accuracy: 0.9679
Test set:
105/105 - 2s - loss: 0.0041 - accuracy: 0.9991
```

![image](plots/accuracy.png)

![image](plots/loss.png)

## Dataset Splitting
The OAI AKOA dataset was split into 3 subsets - training, validation and test. The data was split on patients rather than individual images to prevent data leakage between sets. Checks were also made to ensure each set had a sufficiently balanced ratio of *'right'* labelled data to *'left'* labelled data (between 0.3 and 0.7) to prevent class imbalance. 

The test subset was taken as 20% of the initial number of patients and the validation subset was then taken as 20% of the remaining number of patients e.g. the split ratios were
* Test = **0.2**
* Validation = (0.2 * (1 - Test)) = **0.16**
* Training = (1 - Test - Validation) = **0.64**

## Authors
Khang Nguyen
