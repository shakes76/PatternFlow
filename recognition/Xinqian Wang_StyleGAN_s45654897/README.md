This is my first style GAN impelementation. Because of short in GPU, the model is intermittently trained on Colab.  
The README file is consisted of two parts, firstly general introduce the Style model and the rest part is some tests I did.  
Name: Xinqian Wang   |   Student_ID: 45654897   |   Update_Time: 17:20/30/10/2021

# MRI Image Gereration By Implementing StyleGan
> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> https://arxiv.org/abs/1812.04948

The code I used is basicly from Kim Seonghyeon's Github.
And thanks for his kindly reply to me for helping me better understand the model.
- Address: https://github.com/rosinality/style-based-gan-pytorch/issues/34

Usage:
```
Image Dataset (type: Folder)
│
└───class1 (type: Folder)
│   │   image01.jpg
│   │   image02.jpg
│   │   image03.jpg
│   │   ...
│   
└───class2 (type: Folder)
│   │   image01.jpg
│   │   image02.jpg
│   │   ...
│
... ...
```
1. Prepare the lmdb dataset, the input **Image Dataset** should have the structure as above
> to be continue
2. train the model
> to be continue
3. prepare the lmdb dataset
> to be continue
***********************************************************************************************************************
## Introduction
### Model Structure
I made a plot for better explain how the class in the file `model.py` construct the StyleGan. The name of the class I used is in the grid tracked by the red line.

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/566d7c282d2fc64490219a38d07c6cf7379591cf/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Model%20Structure.png" alt="" width='80%' height='80%'/>
</p>

### Reference Destruction
For people who would like to know more about the literatures which according to a specific concept in the StyleGan.
#### Max Blur pooling
1. [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)`Max Blur pooling`
2. [Using pre-training can improve model robustness and uncertainty](https://arxiv.org/abs/1901.09960)`robustness`
3. [A rotation and a translation suffice: Fooling cnns with simple transformations](https://openreview.net/forum?id=BJfvknCqFQ)`Max-pooling failed on anti-aliasing`
4. [Why do deep convolutional networks generalize so poorly to small image transformations?](https://arxiv.org/abs/1805.12177)`Max-pooling failed on anti-aliasing`
#### Equalized Learning Rate and Pixel Norm
1. [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196)`Equalized Learning Rate` `Pixel Norm`
2. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)`Kaiming He’s normalization way`
#### Double Backpropagation
1. [A Closer Look at Double Backpropagation](https://arxiv.org/abs/1906.06637)`2019`
2. [Improving generalization performance using double Backpropagation](http://yann.lecun.com/exdb/publis/pdf/drucker-lecun-92.pdf)`1992`
#### Disentanglement
1. [On the emergence of invariance and disentangling in deep representations](https://arxiv.org/abs/1706.01350)
2. [Isolating sources of disentanglement in variational autoencoders](https://arxiv.org/abs/1802.04942)
3. [A framework for the quantitative evaluation of disentangled representations](https://openreview.net/pdf?id=By-7dz-AZ)
4. [A survey of inductive biases for factorial representation-learning](https://arxiv.org/abs/1612.05299)
5. [Learning factorial codes by predictability minimization](https://ieeexplore.ieee.org/document/6795705)
#### Gram Matrix
1. [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036)
2. [Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
#### Exponential Moving Average
1. [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
2. [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406)
3. [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196)
**********************************************************************************************************************
## Model Visulization
### UMAP Visulization
We madke 10 basically same codes with the shape of [,512] and index them from 0 to 9. The only difference between these 10 codes is within the 98th value. We set the code with 0 index's valur in the 98th as the original value v. For the i-th code, the value in the 98th equals to <img src="https://latex.codecogs.com/svg.image?v&space;&plus;&space;i^{3}&space;" title="v + i^{3} " />. Then, we plot the codes which outputed by the Mapping Network using U-MAP function. The plot is as below:

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/86015cf5f42b1d2a07143fbdef9f2b7bdd54333a/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Umap.jpg" alt="" width='70%' height='70%'/>
</p>

For comparison, we also made three plots which attached their generated images. As below:

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/86015cf5f42b1d2a07143fbdef9f2b7bdd54333a/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Maniford_01257.png" alt="" width='70%' height='70%'/>
</p>

**Inference:** We can deduce for the index set of 0, 1, 2, the generated image is more bright.

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/86015cf5f42b1d2a07143fbdef9f2b7bdd54333a/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Maniford_123467.png" alt="" width='70%' height='70%'/>
</p>

**Inference:** We can deduce for the index set of 1, 2, 3, 4, 6, 7, the generated image is generally changing. It doesn't seem like we have a model collapse.

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/86015cf5f42b1d2a07143fbdef9f2b7bdd54333a/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Maniford_56789.png" alt="" width='70%' height='70%'/>
</p>

**Inference:** We can deduce for the index set of 5, 6, 7, 8, 9, the generated image is very similar to each other. Additionally, it appears that the black area in the middle of the brain tends to be larger if the point is closed to the y-axis.

## How the mapping network dis-entangle the random tensor?
### Search for distribution
We first generate 4 sets of random vectors with the dimension of (2000,512) by using 3 different ways, torch.randn(Normal Distribution), torch.rand(Uniform Distribution), and torch.randint(Uniform Distribution with integers). Then, we compared the latent code z (z1,z2,z3,z4) before going into the mapping network with the intermediate latent code w (w1,w2,w3,w4) about their means and standard deviation as the table below:

```python
#The data typr transformation will not be provided in this script, just basic ideas
z1 = torch.randn(2000, 512).to(device)
z2 = torch.rand(2000, 512).to(device)
z3 = torch.randint(200,400,(2000, 512),dtype=torch.float32).to(device)
z4 = torch.randint(10000,80000,(2000, 512),dtype=torch.float32).to(device)

w1,w2,w3,w4 = mapNet(z1), mapNet(z2), mapNet(z3), mapNet(z4)
```

| W  | W-mean | W-std |
| -- | ---- | --- |
| w1 | 0.0043377355  | 0.026536208  |
| w2 | 0.0044174353  | 0.027271809  |
| w3 | 0.0045437375  | 0.02691137   |
| w4 | 0.004477056   | 0.027103048  |

| Z  | Z-mean | Z-std |
| -- | ---- | --- |
| z1 | 9.3621886e-05  | 0.99944264  |
| z2 | 0.5001404  | 0.28848535  |
| z3 | 299.55765  | 57.77888   |
| z4 | 45018.316   | 20201.037  |

**Inference:** We learned that one of the Mapping Network's function is to transform the random vector to a particular distribution with fixed mean and standard deviation.

We also made two plots, with the number of 500 x,y sets which represents mean,std. For each set, these mean and std are calculated from a number of 2000 vectors we made. The plot on the left is for the latent code z and right is for the intermediate latent code w.

Latent code (Z)         |  Intermediate latent code (W)
:-------------------------:|:-------------------------:
![](https://github.com/Wangxinqian/PatternFlow/blob/413216e5e6a31c9ebf87b7cc1f87f8f0fe0860b8/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/w_mean_std.png)  |  ![](https://github.com/Wangxinqian/PatternFlow/blob/413216e5e6a31c9ebf87b7cc1f87f8f0fe0860b8/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/z_mean_std.png)

**Inference:** We can see for the latent vector without going into the mapping network, it takes up most of the plot. However, for the intermediate latent code coming after the mapping network, it presents a linear format, when the mean higher, the std is ten to go increase too.

### Search for connection between different columns
Another way of analyze the mapping network is to map plots with two different random pick columns from the row space and compared with Z and W. Below, we generated a shape of (4000,512) random code Z and get W by let Z pass through the mapping network. We randomly compared two different column inside Z or W. The code below cler describe what we did.

```python
z = torch.randn(4000, 512).to(device)
w = mapNet(z)

a1,a2 = w[:,2],w[:,118]

fig = plt.figure()
ax = plt.subplot()
ax.scatter(a1, a2, c='red')

plt.xlabel('column #{2}')
plt.ylabel('column #{118}')
plt.show()
```

Latent code (Z)         |  Intermediate latent code (W)
:-------------------------:|:-------------------------:
![](https://github.com/Wangxinqian/PatternFlow/blob/bf454b40502fc6a32cd3923525341d15f440927c/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/z_c_188.c_481.png)  |  ![](https://github.com/Wangxinqian/PatternFlow/blob/bf454b40502fc6a32cd3923525341d15f440927c/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/w_c_188.c_481.png)

**Inference:** We can see for the plot on the right, you barely can see any patterns inside the plot. However, the plot on the left you can observe some rules. This sidely proves how the mapping network's operating during the training. The future could also made by analyzing the covarience matrix by looking for the relations between different attributes.

## How is the style mixing (Section 3.1) performance?
### Two sets of images were generated from their respective latent codes (Source A and B)
Source A, is on the **Up** side. Source B, is on the **left** side.
We set the Source A as the input style for the first two different resolutions, 8×8 and 16×16, and the rest resolutions are all inputed by Source B.
The outcome is provided as a form of image, you can see below, we also printed it's mean and standard deviation in different colors for each image.
<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/06a2beec098afadef6b3466f55d5353acdc2c2fa/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/mix_SourceA_SourceB.png" alt="" width='70%' height='70%'/>
</p>

**Inference:** The small resolution determines how the image looks like. Whereas, the higher resolution may deside some tiny thing. It may not be so obvious in the image we provide, but we also observe the consistency of the brightness is decided by higher resolution latent code (Source B).

## Stochastic variation (Section 3.2), diving into how the input noise affect the generated image
### Stochastic variation example-1
We generate three images with different noise set. We plot the image below. And we can see a very tiny slightly change when comparing the images with each other. They all got same shape and you can hardly find out differences between them.
<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/6f06e7be4c74bf5b51bfe6ca577849c7a0a72739/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/noise_variation.png" alt="" width='70%' height='70%'/>
</p>

### Stochastic variation example-2
We also generated 2 pairs of images. The first one is the comparation between zero noise and noise normally injected into all layers.
<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/6f06e7be4c74bf5b51bfe6ca577849c7a0a72739/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/noise%20and%20zero%20noise.png" alt="" width='60%' height='60%'/>
</p>

Next, we divided our model into two parts as with coarse layers and fine layers correspont to different resolution. Considering our model's highest resolution is onl 256. We define the coarse layers is refered to the layers which gets the resolution of 8 and 16. The rest all is defined as fine layers.
Below is the comparation between noise only injected into coarse layers and noise only injected into fine layers

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/b0ec278a16f0f16aaad55368bcb7fae39eefdd26/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/noise%20coarse%20and%20fine_2.0.png" alt="" width='60%' height='60%'/>
</p>

**Inference:** The conclusion is same as the paper mentioned. The noise tends to control the texture of the image. If noise the more likely to turn the image into a paint-like style. Additionaly, the noise is used by the generator to do some decorations for changing the images' details so that the generator can make different which has different variation. Or, we can deduce, the main shape of the image is not sensitive to the noise whereas the details of image are very sensitive to the noise.

> The previous sections as well as the accompanying video demonstrate that while changes to the style have global effects (changing pose, identity, etc.), the noise affects only inconsequential stochastic variation (differently combed hair, beard, etc.).


## The difference of Gram matrix and Channel-wise Mean in the style-mixing
### What's Gram Matrix
Gram Matrix is formed by the inner product of two matrixes. Intuitively, if the inner product is greater than 0, then these two matrixes are basically having same direction in the vetor space. If their inner product equals to 0, they are orthogonal. Otherwise, their direction is not homogeneous.

For the image case, imaging there are two images, A and B, and they have the same structure in a shape of [C,H,W]. Then, change A and B's shape into [C, H*W]. Finally, we got A's Gram Matrix by doing inner product [C, H*M] * [H*W, C]. The difference between their style is just calculate the MSE loss.

We compute the MSE loss between different Gram Matrixes. The image below is the style mixing picture which is similar as Source A(Up) and Source B(Left) talked about befroe. And we indexed these pictures for better describing the picture. For example (1,1) refers to the grey picture on the up-left.

<p align="center">
  <img src="https://github.com/Wangxinqian/PatternFlow/blob/3541b867cfacb313e9e96baa8f509230416e0568/recognition/Xinqian%20Wang_StyleGAN_s45654897/image/Gram%20Matrix%20plot_New.png" alt="" width='30%' height='30%'/>
</p>

The image on the position (2,2) are images generated from Source A inputed from the resolution of 8×8 and 16×16 and Source B inputed from the rest resolution layers. Then we start computing the MSE Distance between two images' Gram Matrixes. In the case [image (2,1)] is generated from Source A-> [image (1,2)] and Source B-> [image (2,1)].

The MSE between between [image (1,2)]'s Gram Marix and [image (2,1)]'s Gram Marix is 0.0003.
The MSE between between [image (1,2)]'s Gram Marix and [image (2,2)]'s Gram Marix is 0.0002.
The MSE between between [image (2,1)]'s Gram Marix and [image (2,2)]'s Gram Marix is 1.1183e-05.

We done its trial for a hundred times, the case where the MSE loss between [image (1,2)] and [image (2,2)] is greater than the MSE loss between [image (2,1)] and [image (2,2)] happends with a probability of 87.6%.


**Inference:** In conlusion, we deduce a image in the generation process would tend to be similar in shape with the style from coarse spatial resolutions. However, in terms of mean and MSE distence between Gram Matrix, it tends to be closed to the style inputed from higher resolutions.
