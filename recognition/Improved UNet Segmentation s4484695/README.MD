# Improved 2D-UNet ISIC2017

## Algorithm Description

![Improved UNet Architecture](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/UNet.PNG)

This UNet is implemented in order to solve the problem of segmenting lesions from ISIC2017 Data Set. Metric used to determine performance is Binary Dice Coefficient.

This algorithm passess the input through context layers, this is done to extract the most relevant features from the input. Once the input has been passed through these context layers it is passed back up through localization layers to construct an output. Stride 2 convolutions are used on the input during contextualization in order to downsize the input. Batch dropout coefficient of 0.3 is employed during contextualization to mitigate the risk of overfitting. Skip connections are used going from context layer **n** to localisation layer **n**, this allows for fine grained details to be recovered. During localization pathway, integration of segmentation layers at various levels of the localization pathway leads to a better output of the network.

The input is passed through a sigmoid layer. While, in the referenced paper (“Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,”) a softmax layer is employed. I've chosen to use a sigmoid layer since the output is binary. Various hyper parameters have also been tweaked when compared to the paper such as epoch number reduced from 300 to 30, this was done because the ISIC2017 dataset is easier to solve. Please note I used a binary dice loss and binary dice coefficient for this implementation since the output from my model is binary.

## How to Use Model

Ensuring that you have the appropriate ISIC2017 DataSet and is split into folders as described in train.py comments.

In train.py and predict.py replace placeholders with appropriate paths.
1. Run train.py, wait until training is completed will recieve a print in console similar to:

![Train.py Example](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/trainconsole.PNG)

2. Run predict.py, making sure not to delete or move model file created by train.py. Once inference is complete will recieve a print in console similar to

![predict.py Example](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/testconsole.PNG)

3. Navigate to your specified path, where model, train.py and predict.py are stored. You will find .png files with data vizualisations.

## Dependencies

- Python = 3.9.13
- PyTorch = 1.12.1
- TorchVision = 0.13.1
- NumPy = 1.23.1
- CudaToolKit = 11.6.0
- tqdm = 4.64.1
- MatPlotLib = 3.5.2

## Reproducibility of Results

Use of shuffle=True on Training DataLoader means that model could recieve different sequence of inputs during training. Weight Initialization for parameters is set to default and so parameter weight initialization will always be same. Due to shuffle on training dataloader a different minimum on manifold could be found, and therefore cannot gaurantee reproducibility during training.

Assuming that two users load the same model state dictionary during predict.py execution then test inference is reproducible given they use the same dataset in same order, since shuffle=False for Test Dataloader.

## Pre-Processing and Data Splits

ISIC2017 is pre-split upon download into training, validation and testing. Training set has 2000 images and labels, validation set has 150 images and labels, test set has 600 images and labels.

Only pre-processing done to images and labels is conversion from PIL Image to tensor using ToTensor(), and resizing to 128x128. Resizing reduces the time taken to train and infer.

## Training and Validation Binary loss and Dice Coefficient Plots

![Dice Coefficient vs Epochs](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/DiceCurve.png)

![Dice Loss vs Epochs](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/LossCurve.png)

## Training and Validation Visualizations

### Epoch 1

![Epoch 1](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/Epoch1.PNG)
![Epoch 1](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch1.png)

### Epoch 5

![Epoch 5](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/Epoch5.PNG)
![Epoch 5](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch5.png)

### Epoch 30

![Epoch 30](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/Epoch30.PNG)
![Epoch 30](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/ValidationSegmentsEpoch30.png)


## Test Inference Vizualisation

![Test Inference Vizualisation](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/Inference.PNG)
![Test Inference Vizualisation](https://github.com/Kamran496/PatternFlow/blob/master/recognition/Improved%20UNet%20Segmentation%20s4484695/Results/TestSegments.png)

## References
- https://arxiv.org/abs/1802.10508v1
- https://github.com/pykao/Modified-3D-UNet-Pytorch
- https://colab.research.google.com/drive/1nfXJrNN3Q6yQMrtldeHx7wx7QuO1YsDQ?usp=sharing#scrollTo=3YoDJJZjxV6B
- https://github.com/pykao/Modified-3D-UNet-Pytorch
- https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html#:~:text=mean%3A%20simply%20divide%20the%20sum,(df)%20*%20image_size%20*%20image_size
- https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
