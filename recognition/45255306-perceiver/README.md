# Perceiver: General Perception with Iterative Attention
Perceiver is a transformer based model that is designed to handle data of different modalities including image, video, audio, etc. Perceiver utilizes 2 stage attention layer which are cross attention and latent attention.The issue with multimodal models utilizing transformers as their base architecture is the quadratic complexity of the attention algorithm. Image data such as ImageNet has a dimension of 224 x 224 = 50176, this is a really huge number to perform attention, hence, computationally expensive to do. Perceiver tackles this issue by utilizing cross attention on a fixed dimension latent bottleneck and the input data, where the dimension of the fixed latent is smaller than the input data.

## Dependencies
The perceiver model utilizes PyTorch and Einops library. The experiment/driver script utilizes PyTorch, glob, tqdm, numpy and pillow/PIL.

## Dataset
The dataset used for this experiment is AKOA knee dataset. It is a classification problem to classify laterality (left or right sided knee).The dataset are seperated to left and right knee MRI images. 

### Right Knee
There are a total of 10560 images for right knee. For the experiment, the right knee data are split as shown below.
* Train - 7920
* Val - 1320
* Test - 1320

### Left Knee
There are a total of 7640 images for left knee. For the experiment, the left knee data are split as shown below.
* Train - 6040
* Val - 800
* Test - 800

The original MRI data is a 3D knee nii data, however, it is converted to 2D images by splitting a 3D data into 40 seperate images. In order to prevent data leakage, the 40 images of a single 3D knee needs to be in either train, validation or test dataset. As a result, the following steps are done to split the data.

1. Image data paths of left and right knee are extracted and sorted.
2. The dataset for train, validation and test are split by numbers divisible by 40, to ensure no data leakage.
3. Left and right knee dataset are merged to each respective dataset subset of train, validation and test.

Following these steps prevents data leakage and also ease the experiment for reproducibility.

## Model usage
```python
import torch
from perceiver import Perceiver

model  = Perceiver(6, 10., 6, input_channels=1, input_axis=2,
            num_latents=512, latent_dim=512, cross_heads=1, latent_heads=8,
            cross_dim_head=64, latent_dim_head=64, num_classes=2, attention_dropout=0.)

img = torch.randn(1, 3, 28, 28)
model(img)
```

## Experiment Results
Several experiments are run utilizing different hyperparameters, latent dimensions and activation function. The paper mentioned the use of GeLU activation function. However, in this experiment, it is shown that GeGLU increases the performance of the model and speeds up the convergence rate.

The hyperparameter used for the best results is:

Model architecture:
* num_freq_bands - 6
* max_freq - 10 
* depth - 6
* number of latents - 512
* latent_dim - 512 
* cross_heads - 1 
* latent_heads - 8
* cross_dim_head - 64 
* latent dimension head - 64
* attention dropout - 0

Experiment setup
* Optimizer - Adam (Learning Rate: 2e-4)
* Criterion - CrossEntropyLoss
* Scheduler - ExponentialLR with decay rate of 0.995
* Batch Size - 64

#### Accuracy and Loss graph
![Perceiver with GeGlu](https://user-images.githubusercontent.com/67994195/139537574-f8965638-49f2-4538-9c04-9d8c3f6f94ae.png)

The experiment saves the best validation loss model, the best model has following result:
|            | Accuracy | Loss   |
|------------|----------|--------|
| Train      | 93.8%    | 0.1822 |
| Validation | 94.9%    | 0.1348 |

#### Confusion Matrix For Test Dataset
![Test Confusion Matrix](https://user-images.githubusercontent.com/67994195/139537568-94e4c2ba-ecdd-464f-bdb0-eb06371bf812.png)

The best model has a test accuracy of 92.6%.
