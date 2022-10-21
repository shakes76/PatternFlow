# Improved UNET - Segmentation of ISIC Dataset
Paragraph about ISIC dataset and what this project is

## ISIC Dataset

Available: https://challenge.isic-archive.com/data/#2017

## Improved UNET Model Architecture

### Context Module


### Upsampling Module


### Localisation Module


### Segmentation Layer

## Training

### Optimiser and Loss Function

### Training Accuracy and Loss

### Model Adjustments
Initially, the structure layed out in the paper was followed for model training. However, the model test accuracy didn't have a Dice similarity coefficient of greater than 0.8.
![Test data dice coefficient original](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientOriginal.JPG)
First, the learning rate scheduler which was used as per the paper was removed.
'''
results = model.fit(train_batch, epochs=self.epochs, validation_data=validate_batch,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                lambda epoch: self.learning_rate * 0.985 ** (epoch)
                            )])
'''
Replaced with a constant learning rate.
'''
adamOptimizer = Adam(learning_rate=self.learning_rate)
model.compile(optimizer=adamOptimizer, loss=self.diceLoss, metrics=[self.diceCoefficient])
'''
This resulted in a slight increase in the dice similarity coefficient.
![Test data dice coefficient adjust learning rate](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientLearningRate.JPG)
By decreaseing the training batch size the model was able to achieve a dice similarity coefficient greater than 0.8 for the test set.
![Test data dice coefficient adjust batch size](https://github.com/ryanjohnson08/45307915/blob/topic-recognition/recognition/45307915/images/TestDataDiceCoefficientDecreaseBatchSize.JPG)
## Predictions

## Usage

### Dependencies
- Python: 3.7.15
- Numpy: 1.21.6
- Matplotlib: 3.2.2
- Tensorflow: 2.9.2
- Tensorflow Addons: 0.18.0

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
