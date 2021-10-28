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

## Implementation 

The model and network code written in Pytorch follow the [official Tenforflow implementation](https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py) very carefully. Zhuoxiao rewrote and reproduced the code using PyTorch according to the official Tensorflow code. Thus, the code right should belong to Copyright (c) 2019, [NVIDIA CORPORATION](https://www.nvidia.com/en-us/).

The implementation can be roughly divided into three steps. The first step (preprocess_AKOA_images.py) is the AKOA pre-processing to convert each AKOA image to all the different dimensions (8,16,32). The second step (train_AKOA_StyleGAN.py) is the training process, which accepts pre-processed AKOA images to train the StyleGAN. The last step (generate_AKOA.py) is to produce fake images using trained StyleGAN. 

#### Hardware

The implementation code should be run at least **40 hours** with more than **60k iterations**, using **two RTX 2080 Ti GPUs**, to get the expected performance as displayed in next section. 

#### Software 


```
'''
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
_libgcc_mutex=0.1=main
_openmp_mutex=4.5=1_gnu
ca-certificates=2021.9.30=h06a4308_1
certifi=2020.12.5=py36h06a4308_0
cycler=0.10.0=pypi_0
kiwisolver=1.3.1=pypi_0
ld_impl_linux-64=2.35.1=h7274673_9
libffi=3.3=he6710b0_2
libgcc-ng=9.3.0=h5101ec6_17
libgomp=9.3.0=h5101ec6_17
libstdcxx-ng=9.3.0=hd4cf53a_17
lmdb=1.2.1=pypi_0
matplotlib=3.3.4=pypi_0
ncurses=6.2=he6710b0_1
numpy=1.19.5=pypi_0
openssl=1.1.1l=h7f8727e_0
pandas=1.1.5=pypi_0
pillow=8.4.0=pypi_0
pip=21.0.1=py36h06a4308_0
pyparsing=2.4.7=pypi_0
python=3.6.13=h12debd9_1
python-dateutil=2.8.2=pypi_0
pytz=2021.3=pypi_0
readline=8.1=h27cfd23_0
setuptools=58.0.4=py36h06a4308_0
six=1.16.0=pypi_0
sqlite=3.36.0=hc218d9a_0
tk=8.6.11=h1ccaba5_0
torch=1.2.0=pypi_0
torchvision=0.4.0=pypi_0
tqdm=4.62.3=pypi_0
wheel=0.37.0=pyhd3eb1b0_1
xz=5.2.5=h7b6447c_0
zlib=1.2.11=h7b6447c_3
```

#### Pre-process the AKOA Dataset Using preprocess_AKOA_images.py

Since the progressive StyleGAN is built in this project. All AKOA images should be trained progressively. For example, all images are trained at dimension 8 times 8 for one phase and 16 times 16 for another phase. Finally, the expected dimension 258 times 258 is trained as the final phase. 

Therefore, the pre-processing is simply converted all the AKOA images into all the different dimensions with key and value pairs. The output file should be stored in a directory containing processed .mdb files. This directory is later used for the training path.

To get started with the preprocess_AKOA_images.py, user can read the parameter information by:

```
python preprocess_AKOA_images.py --help
```

And the command line above should print very detailed information for each parameter:

```
usage: preprocess_AKOA_images.py [-h] [--number_worker NUMBER_WORKER]
                                 [--output_directory OUTPUT_DIRECTORY]
                                 AKOA_raw_images_path

Pre-process the AKOA dataset for further training the Progressive StyleGAN.

positional arguments:
  AKOA_raw_images_path  specify the path of the AKOA raw images

optional arguments:
  -h, --help            show this help message and exit
  --number_worker NUMBER_WORKER
                        specify number of cpus to be used for pre-process the
                        AKOA datasets
  --output_directory OUTPUT_DIRECTORY
                        specify the output directory to store the pre-
                        processed .mbd files for later training
```

#### Train StyleGAN using train_AKOA_StyleGAN.py

To train the StyleGAN using AKOA dataset, some of parameters of train_AKOA_StyleGAN.py should be specified. To check the parameter information and its usage, users can simply input the code below: 

```
python train_AKOA_StyleGAN.py --help
```

And the command line above should print very detailed information for each parameter:

```
usage: train_AKOA_StyleGAN.py [-h] [--start_dimension START_DIMENSION]
                              [--final_dimension FINAL_DIMENSION]
                              [--progressive_stage PROGRESSIVE_STAGE]
                              [--loss_method {WGAN}] [--mixing] [--ckpt CKPT]
                              [--lr LR]
                              AKOA_directory

Training StyleGAN for generating StyleGAN

positional arguments:
  AKOA_directory        The directory of AKOA dataset must be specified for
                        training the StyleGAN.

optional arguments:
  -h, --help            show this help message and exit
  --start_dimension START_DIMENSION
                        specify the start dimenstion of image for progressive
                        GAN training.
  --final_dimension FINAL_DIMENSION
                        specify the final(max) dimenstion of image for
                        progressive GAN training. The final_dimension defines
                        the max dimenstion of the image that can be generated
                        by the generator.
  --progressive_stage PROGRESSIVE_STAGE
                        images to be trained for each progressive_stage, for
                        example, 1 state could be training for 16*16
                        resolution feature map.
  --loss_method {WGAN}  Define the loss function. In AKOA datasets, WGAN loss
                        is sufficient.
  --mixing              apply the mixing module as proposed in the paper
  --ckpt CKPT           specify the checkpoints to resume the model training.
  --lr LR               specify the learning rate used for StyleGAN training.
```


#### Generate Fake AKOA Images using generate_AKOA.py

Once the StyleGAN is fully trained, the checkpoint containing all the parameters needed for generating fake AKOA images should be saved.

Then, the generate_AKOA.py script can be called to generate fake AKOA images. To know how to utilise the generate_AKOA.py images, please read the parameter description carefully by inputting:

```
python generate_AKOA.py --help
```

And the command line above should print very detailed information for each parameter:

```
usage: generate_AKOA.py [-h] [--column_size COLUMN_SIZE] [--row_size ROW_SIZE]
                        [--dimension DIMENSION]
                        checkpoint

Generate fake AKOA images using trained generator.

positional arguments:
  checkpoint            checkpoint to checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --column_size COLUMN_SIZE
                        number of columns of sample matrix
  --row_size ROW_SIZE   number of rows of sample matrix
  --dimension DIMENSION
                        the dimension of the generated AKOA image
```

## Result and Analysis
Generated AKOA Images


<img src="https://user-images.githubusercontent.com/50613939/138634887-9fb01e51-ca6a-4dd8-aa2f-641a47424ae5.png" alt="drawing" width="800"/>
<img src="https://user-images.githubusercontent.com/50613939/138634910-76afdf8c-29c4-4799-a860-955871c7454a.png" alt="drawing" width="800"/>


## Conconlution
Finally...
