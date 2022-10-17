# COMP3710-report
## Dependencies
- tensorflow==2.9.1
- numpy==1.23.1
- matplotlib==3.6.1
## Description of Improved Unet
### What is Improved Unet
Improved-Unet were first introduced in F. Isensee, P. Kickingereder, W. Wick,
M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival 
Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available:
https://arxiv.org/abs/1802.10508v1

In the essay, author made an improvement based on classic Unet structure when dealing with 
BraTS 2017 challenge. To get the segmentation of given images, they came up with the idea 
that design a 3D input/output neuron network to get different features by different layers.


### Structure
In encoder part, a method called context module is used five times to extract the features.
combined with several 3 * 3 * 3 stride 2 convolution layers.
In decoder part, Upsampling Layers were used in order to prevent checkerboard artifacts in 
network output.

Improved-Unet has similar function as normal Unet, it can get the segmentation of biomedical
image.
One of the most important feature of Unet is putting relatively early data(in encoder part) 
into late data(in decoder part). To make them combine, concatenate layers are used.
! [ Structure ](https://github.com/TianyuWu-UQ/PatternFlow/blob/topic-recognition/recognition/Improved_unet_ISIC/structure.png)

## ISIC dataset
The dataset used in my implement is based on the ISIC dataset which contain images of different
skin diseases.
Because of hardware limitations, I used:
- 1051 image pairs contains origin data and segmentation image for training.
- 110 image pairs for validation.
- 219 images for get the output of Improved Unet model.
! [ input_images ](https://github.com/TianyuWu-UQ/PatternFlow/topic-recognition/recognition/Improved_unet_ISIC/input_example.jpg)

## Files contained
- `dataset.py` containing the data loader, in which dataset is first being selected and useless
images are deleted(superpixels image), then load the data and stack each image 16 times to get
the 3D dataset. The shape of images collected is 64 * 64, not same as 128 * 128 in the essay
 according to hardware limitation. 
- `modules.py` containing the model constructor. All components of Improved Unet are included.
- `train.py` containing the trainer. A learning_rate_schedule is used to get an Exponential Decay.
compare to the Dice loss function used in the reference essay, I choose using typical categorical
cross entropy to get a more accurate output after some attempt with dice loss as loss function.
- `predict.py` containing the source code to get the result using a trained model with test dataset.

## Model and training result 
When training the model, some statistics are used according to the reference essay. I set the batch_size 
of each epoch to 100 and run 300 epochs in total. As hardware limitation, after implement source code in 
PyCharm IDE, I use Google Colab to run the code and train the model. Part of training process are listed below.

First 5 epochs:
```
Epoch 1/300
11/11 [==============================] - 4s 388ms/step - loss: 0.6794 - accuracy: 0.6937 - val_loss: 0.6354 - val_accuracy: 0.7395
Epoch 2/300
11/11 [==============================] - 4s 385ms/step - loss: 0.5356 - accuracy: 0.8024 - val_loss: 0.3658 - val_accuracy: 0.8967
Epoch 3/300
11/11 [==============================] - 4s 387ms/step - loss: 0.4810 - accuracy: 0.8204 - val_loss: 0.4090 - val_accuracy: 0.8275
Epoch 4/300
11/11 [==============================] - 4s 384ms/step - loss: 0.3650 - accuracy: 0.8583 - val_loss: 0.3628 - val_accuracy: 0.8512
Epoch 5/300
11/11 [==============================] - 4s 386ms/step - loss: 0.3166 - accuracy: 0.8761 - val_loss: 0.3181 - val_accuracy: 0.8605

```
Last 5 epochs:
```
Epoch 296/300
11/11 [==============================] - 4s 384ms/step - loss: 0.0193 - accuracy: 0.9918 - val_loss: 0.7602 - val_accuracy: 0.9236
Epoch 297/300
11/11 [==============================] - 4s 382ms/step - loss: 0.0193 - accuracy: 0.9918 - val_loss: 0.7558 - val_accuracy: 0.9236
Epoch 298/300
11/11 [==============================] - 4s 381ms/step - loss: 0.0192 - accuracy: 0.9918 - val_loss: 0.7491 - val_accuracy: 0.9235
Epoch 299/300
11/11 [==============================] - 4s 381ms/step - loss: 0.0194 - accuracy: 0.9918 - val_loss: 0.7466 - val_accuracy: 0.9244
Epoch 300/300
11/11 [==============================] - 4s 381ms/step - loss: 0.0193 - accuracy: 0.9918 - val_loss: 0.7376 - val_accuracy: 0.9248

```

This is result visualisation of model output.
Because there are only two possible output colors(black and white), so the output of model has 2 color channels.
To visualise it, I add an empty third channel to plot it with RGB format, so the color of output visualisation 
is in red and green.
! [result](https://github.com/TianyuWu-UQ/PatternFlow/topic-recognition/recognition/Improved_unet_ISIC/result.png)
A dice similarity coefficient is calculated by formula and get the result of higher than 85% which achieve 
the target.

