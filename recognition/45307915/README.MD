# Improved UNET - Segmentation of ISIC Dataset
Paragraph about ISIC dataset and what this project is

## ISIC Dataset
The ISIC dataset contains a set of skin lesion images and their respective segmentation masks. The 2017 dataset used was already split into training, testing and validation sets.

Due to all the images in the set being different sizes they were all resized. The skin lesion images were resized to (256, 256, 3), keeping their colour channels. The segmentation masks were resized to (256, 256, 1), reducing to a single colour channel. Thresholding was then completed on the segmentation masks, pixels with a value >0.5 were set to 1, and all other pixels set to 0.

![Training Image](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/image.png)
![Training Mask](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/mask.png)

Available: https://challenge.isic-archive.com/data/#2017

## Improved UNET Model Architecture

![Improved UNET Model Architecture](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/IUNET_Architecture.JPG)

Like the original UNET the Improved UNET is made up of encoding and decoding layers. Components of the improved UNET Architecture are as follows.

- Leaky ReLU - "leaky ReLU nonlinearities with a negative slope of $10^{−2}$ for all feature map computing convolutions" [1]
- Instance Normalisation - "replace the traditional batch with instance normalization" [1]
- Context Module - "Each context module is in fact a pre-activation residual block with two 3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between." [1]
- Upsampling Module - "This is achieved by first upsampling the low resolution feature maps, which is done by means of a simple upscale that repeats the feature voxels twice in each spatial dimension, followed by a 3x3x3 convolution" [1]
- Localisation Module - "A localization module consists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the number of feature maps." [1]
- Skip Connection - Like the original UNET the improved UNET makes use of skip connections
- Segmentation Layer - These are 3x3 convolutions with a single output filter. These segments are summed to form the final network output.

## Training

### Optimiser and Loss Function
The optimiser used for training was "the adam optimizer with an initial learning rate $lr_{init} = 5 · 10^{−4}$, the following learning rate schedule: $lr_{init} · 0.985epoch$".

For the loss function the dice coefficient for the true and predicted values was used.

```
def diceCoefficient(self, y_true, y_pred):
        """
        Dice Coefficient

        Credit Karan Jakhar
        https://karan-jakhar.medium.com/100-days-of-code-day-7-84e4918cb72c

        Parameters:
            y_true (tf.Tensor): true output
            y_true (tf.Tensor): output predicted by model

        Return:
            tf.Tensor: Dice coefficient based on true output and prediction

        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
        return dice
```

### Training Dice Coefficient and Loss
The following graphs show the training Dice Coefficient Loss and Accuracy over 20 Epochs.
![Dice Coefficient Loss](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/diceCoefficientLoss.png)
![Dice Coefficient](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/diceCoefficient.png)

### Adjustments
Initially, the structure layed out in the paper was followed for model training. However, the model test accuracy didn't have a Dice similarity coefficient of greater than 0.8.

![Test data dice coefficient original](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientOriginal.JPG)

First, the learning rate scheduler which was used as per the paper was removed.
```
results = model.fit(train_batch, epochs=self.epochs, validation_data=validate_batch,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                lambda epoch: self.learning_rate * 0.985 ** (epoch)
                            )])
```
Replaced with a constant learning rate.
```
adamOptimizer = Adam(learning_rate=self.learning_rate)
model.compile(optimizer=adamOptimizer, loss=self.diceLoss, metrics=[self.diceCoefficient])
```
This resulted in a slight increase in the dice similarity coefficient.
![Test data dice coefficient adjust learning rate](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientLearningRate.JPG)

By decreasing the batch size the model was able to achieve a dice similarity coefficient greater than 0.8 for the test set.
![Test data dice coefficient adjust batch size](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientDecreaseBatchSize.JPG)

## Predictions
Using the model the following segmentation predictions were made from test set inputs.
![prediction1](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p1.png)
![prediction2](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p2.png)
![prediction3](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p3.png)
![prediction4](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p4.png)
![prediction5](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p5.png)
![prediction6](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p6.png)
![prediction7](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p7.png)
![prediction8](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p8.png)
![prediction9](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p9.png)
![prediction10](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p10.png)
![prediction11](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/p11.png)

## Usage

### Dependencies
- Python: 3.7.15
- Numpy: 1.21.6
- Matplotlib: 3.2.2
- Tensorflow: 2.9.2
- Tensorflow Addons: 0.18.0

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
