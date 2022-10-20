# Improved 2D-UNet ISIC2017

## Algorithm Description

![Improved UNet Architecture](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/UNet.PNG)

This algorithm passess the input through context layers, this is done to extract the most relevant features from the input. Once the input has been passed through these context layers it is passed back up through localization layers to construct an output. Stride 2 convolutions are used on the input during contextualization in order to downsize the input. Batch dropouts dropout coefficient of 0.3 are employed during contextualization to mitigate the risk of overfitting. Skip connections are used going from context layer n to localisation layer n, this allows for fine grained details to be recovered. During localization pathways, integration of segmentation layers at various levels of the localization pathway leads to a better output of the network. 

Finally, the input is passed through a sigmoid layer. While, in the referenced paper (“Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,”) a softmax layer is employed. I've chosen to use a sigmoid layer since the output is binary. Various hyper parameters have also been tweaked when compared to the paper such as epoch number reduced from 300 to 30, this was done because the ISIC2017 dataset is easier to solve.

Please note I used a binary dice loss and binary dice coefficient for this implementation since my output from my model is binary.

## Dependencies



## Reproducibility of Results


## Training and Validation Binary loss and Dice Coefficient

### Dice Coefficients vs Epochs
![Dice Coefficient vs Epochs](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/DiceCurve.png)

### Dice Loss vs Epochs
![Dice Loss vs Epochs](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/LossCurve.png)

## Training and Validation Visualizations

### Epoch 1

![Epoch 1](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch1.png)
![Epoch 1](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch1.png)

### Epoch 5

![Epoch 5](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch5.png)
![Epoch 5](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch5.png)

### Epoch 30

![Epoch 30](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch30.png)
![Epoch 30](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch30.png)


## Test Inference Vizualisation

![Test Inference Vizualisation](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/TestSegments.png)
![Test Inference Vizualisation](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/TestSegments.png)