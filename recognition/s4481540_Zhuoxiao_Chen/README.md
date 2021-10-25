# Image Generation for OAI AKOA knee data using StyleGAN
Authour: [Zhuoxiao Chen](https://zhuoxiao-chen.github.io)

Student ID: 44815404

[School of Information Technology and Electrical Engineering (ITEE)](https://itee.uq.edu.au/)

[The University of Queensland](https://www.uq.edu.au/)

## Introduction
This [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) project in semester 2, 2021, is to build one of the most popular Generative Adversarial Network (GAN) - [StyleGAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf), for generating images of [OAI AKOA knee datasets](https://nda.nih.gov/oai/). StyleGAN was published in [CVPR 2019](https://cvpr2019.thecvf.com/), which is the top conference in the computer vision field. StyleGAN was originally designed to generate human faces with very high quality and variation. However, in this project, we utilise the StyleGAN for generating interesting medical imaging - OAI Accelerated Osteoarthritis knee (AKOA) datasets. Osteoarthritis in adults is one of the most common causes of disability. The OAI's objectives are to provide resources that help people better comprehend knee osteoarthritis prevention and treatment. Thus, generated AKOA images with good quality are likely to be helpful for medical researchers and doctors since they may find different but valuable insights from generated images. 

This project report includes mainly three sections: code implementation, methodology principles explanation, the experiment. The code is given with fully detailed inline comments in this repository. The methodology will talk about how the algorithm of StyleGAN works and why StyleGAN can achieve state-of-the-art results. In terms of the experiment, the experimental setting and implementation details will be described in detail. Also, the generated images will be presented at the end of this report.

## Methodology

### Standard GAN
Give some introduction to the standard GAN here. 

### StyleGAN
The motivation of StyleGAN is to find effective methods to control the image synthesis process by inserting latent code into each intermediate convolution later to control the image features at a different size. 

<img src="https://user-images.githubusercontent.com/50613939/138625217-6e2b5ce6-8f7e-4089-8e94-60ff601c1358.png" alt="drawing" width="600"/>


As illustrated in the figure above, the latent code is usually inputted at the begining of the generater of the GAN. However, in StyleGAN, the beginning of the input is just a 4 times 4 times 512 constant. But there is another branch where the latent input code should be input. That branch is a  mapping network containing eight layers of multilayer perceptrons (MLP). Next, at each convolutional block. the output code from MLPs are passed into a learned affine transform followed by the adaptive instance normalisation (AdaIN), represented by 'A' in the figure. 'B' is used to scale the input noise before inserting the noise to each convolutional layer.  The output code from the 8-layer MLP brach is transformed into styles by the affine transformations. Abd the styles are inputted into the AdaIN operation, with the aim to normalise the feature map to zero mean and unit variance before scaling the feature map twice at each convolutional layer. As for the input noise code at each layer before the AdaIN operation, they are designed to generate stochastic features for the synthesis process.  In summary, the generator consists of several convolutional bocks to scale up the feature map. Each convolutional bock has two AdaIN components that accept affine transformed output from 8-layer MLP and noise code. 

To control image synthesis, the styles from 8-layer MLPs can be modified or selected to change a small proportion of the feature or aspects of the final generated image. Each style at the intermediate layer can only adjust a small part of the feature but not affect the consistently varying and learning feature in later convolutional blacks due to the inserted noise for that intermediate layer only. As a result, each style at one intermediate convolution layer can only adjust and reflect the feature learned at that layer only. Then the features are overridden by the AdaIN followed by that convolution layer.


## OAI AKOA knee dataset
Introduce the knee dataset here

## Implementation Details
Some Implementation details here


## Result and Analysis
Put some results here

## Conconlution
Finally...
