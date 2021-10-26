# Image Generation for OAI AKOA knee data using StyleGAN
Authour: [Zhuoxiao Chen](https://zhuoxiao-chen.github.io)

Student ID: 44815404

Email: zhuoxiao \[dot\] chen \[dot\] uq \[dot\] edu \[dot\] au

[School of Information Technology and Electrical Engineering (ITEE)](https://itee.uq.edu.au/)

[The University of Queensland](https://www.uq.edu.au/)

## Introduction
This [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) project in semester 2, 2021, is to build one of the most popular Generative Adversarial Network (GAN) - [StyleGAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf), for generating images of [OAI AKOA knee datasets](https://nda.nih.gov/oai/). StyleGAN was published in [CVPR 2019](https://cvpr2019.thecvf.com/), which is the top conference in the computer vision field. StyleGAN was originally designed to generate human faces with very high quality and variation. However, in this project, we utilise the StyleGAN for generating interesting medical imaging - OAI Accelerated Osteoarthritis knee (AKOA) datasets. Osteoarthritis in adults is one of the most common causes of disability. The OAI's objectives are to provide resources that help people better comprehend knee osteoarthritis prevention and treatment. Thus, generated AKOA images with good quality are likely to be helpful for medical researchers and doctors since they may find different but valuable insights from generated images. 

This project report includes mainly three sections: code implementation, methodology principles explanation, the experiment. The code is given with fully detailed inline comments in this repository. The methodology will talk about how the algorithm of StyleGAN works and why StyleGAN can achieve state-of-the-art results. In terms of the experiment, the experimental setting and implementation details will be described in detail. Also, the generated images will be presented at the end of this report.

## Methodology

### Standard GAN
The generator's task is to generate 'fake' images that resemble the training images. The discriminator's work is to examine an image and determine whether it is a genuine training image or a generated image. Throughout the training, the generator is continuously attempting to outwit the discriminator by creating increasingly convincing frauds, whereas the discriminator works to improve its characterized mainly and accurately identify real and fake images. The match is balanced whenever the generator generates ideal fakes that appear to come straightforwardly from the training examples, and the discriminator aims to predict with a probability of 50% whether the generator result is real or fake. 

If G is represented is as the generator and D is the discriminator, the loss function for GAN is: 

<img src="https://user-images.githubusercontent.com/50613939/138630901-2ded07f0-7f62-4ebd-9eb0-8fb5537608f6.png" alt="drawing" width="600"/>

### Progressive GAN
The key focus of [Progressive GAN](https://arxiv.org/pdf/1710.10196.pdf) is a training process approach for GANs, where at the beginning, the images are always in low resolutions. However, the resolution of the networks can be slowly raised via equipping them with more convolution layers. StyleGAN is also progressive because there are quite a few benifits of progressive GAN. As there is less category knowledge and fewer styles slightly earlier on, the generation of lower resolution is significantly more steady. By incrementally raising the resolution, a more straightforward ultimate objective: identifying a mapping between latent vectors and images is pursued. In exercise, it allows creating megapixel-scale images with confidence because it stabilises the training. 

<img src="https://user-images.githubusercontent.com/50613939/138632015-a2b95a65-60c7-45d3-a5cb-ba3db8df9891.png" alt="drawing" width="600"/>

### StyleGAN
The motivation of StyleGAN is to find effective methods to control the image synthesis process by inserting latent code into each intermediate convolution later to control the image features at a different size. 

<img src="https://user-images.githubusercontent.com/50613939/138625217-6e2b5ce6-8f7e-4089-8e94-60ff601c1358.png" alt="drawing" width="600"/>


As illustrated in the figure above, the latent code is usually inputted at the begining of the generater of the GAN. However, in StyleGAN, the beginning of the input is just a 4 times 4 times 512 constant. But there is another branch where the latent input code should be input. That branch is a  mapping network containing eight layers of multilayer perceptrons (MLP). Next, at each convolutional block. the output code from MLPs are passed into a learned affine transform followed by the adaptive instance normalisation (AdaIN), represented by 'A' in the figure. 'B' is used to scale the input noise before inserting the noise to each convolutional layer.  The output code from the 8-layer MLP brach is transformed into styles by the affine transformations. Abd the styles are inputted into the AdaIN operation, with the aim to normalise the feature map to zero mean and unit variance before scaling the feature map twice at each convolutional layer. As for the input noise code at each layer before the AdaIN operation, they are designed to generate stochastic features for the synthesis process.  In summary, the generator consists of several convolutional bocks to scale up the feature map. Each convolutional bock has two AdaIN components that accept affine transformed output from 8-layer MLP and noise code. 

#### Control the synthesis
To control image synthesis, the styles from 8-layer MLPs can be modified or selected to change a small proportion of the feature or aspects of the final generated image. Each style at the intermediate layer can only adjust a small part of the feature but not affect the consistently varying and learning feature in later convolutional blacks due to the inserted noise for that intermediate layer only. As a result, each style at one intermediate convolution layer can only adjust and reflect the feature learned at that layer only. Then the features are overridden by the AdaIN followed by that convolution layer.

#### Style mixing
The mixing regularisation attempts to input two branches: different latent codes from 8-layer network, and these two codes are inserted into the generator network randomly when the GAN is trained. The styles should be independent when inserting into the convolutional layers, which is achieved by the normalisation of AdaIN. As discussed in the paragraph above, the features are only controlled by the style at each convolutional layer but not be affected by previous convolutions. However, the styles are always outputted from the same 8-layer mapping network. Thus, the styles may be correlated. Using two different 8-layer networks randomly can ensure that styles at different convolutional layers are not correlated, which means the styles inserted into even adjacent layers are independent.


## OAI AKOA Knee Dataset

The dataset has a size of 1.6GB and contains 18k images of AKOA for training the StyleGAN. 

The images below are randomly collected from AKOA datasets for visualisation. 

![OAI9896743_BaseLine_3de3d1_SAG_3D_DESS_WE_RIGHT nii gz_0](https://user-images.githubusercontent.com/50613939/138839249-657e9785-2e88-43f0-a68c-a89246f91903.png)
![OAI9961728_BaseLine_3_de3d1_SAG_3D_DESS_WE_LEFT nii gz_31](https://user-images.githubusercontent.com/50613939/138833674-52b366a9-a4b3-49c6-8ea8-85a8b50bb182.png)
![OAI9961728_BaseLine_101de3d1_SAG_3D_DESS_WE_LEFT nii gz_4](https://user-images.githubusercontent.com/50613939/138833711-904b3b2a-7ef6-45c4-9f4b-141afdb41285.png)
![OAI9961728_BaseLine_100de3d1_SAG_3D_DESS_WE_LEFT nii gz_8](https://user-images.githubusercontent.com/50613939/138833722-559bcd88-75b2-4c7b-a587-ac8cc6be7738.png)
![OAI9708289_BaseLine_4_de3d1_SAG_3D_DESS_WE_LEFT nii gz_38](https://user-images.githubusercontent.com/50613939/138833787-692e857f-e6e2-420d-817e-cea202d49d70.png)
![OAI9958220_BaseLine_4_de3d1_SAG_3D_DESS_WE_LEFT nii gz_16](https://user-images.githubusercontent.com/50613939/138833738-a9ec071b-9d57-4acc-8dc1-f7a41e4ae5e7.png)

## Implementation Details

The implementation code should be run at least 40 hours with more than 60k iterations, using two RTX 2080 Ti GPUs, to get the expected performance as displayed in next section. 

## Result and Analysis
Generated AKOA Images


<img src="https://user-images.githubusercontent.com/50613939/138634887-9fb01e51-ca6a-4dd8-aa2f-641a47424ae5.png" alt="drawing" width="800"/>
<img src="https://user-images.githubusercontent.com/50613939/138634910-76afdf8c-29c4-4799-a860-955871c7454a.png" alt="drawing" width="800"/>


## Conconlution
Finally...
