# Simple Diffusion Image Generation

Simple diffusion based image generation using PyTorch. This model can learn from a dataset of images and generate new images that are perceptually similar to those in the dataset.
![plot_epoch94](https://user-images.githubusercontent.com/99728921/200819650-d44fd0d3-9529-4c9b-bb1b-3c932a0abf40.jpeg)

#### References

Huge thanks to these videos for helping my understanding:

* [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=912s)
  * This repo was built largely referencing code in this [colab notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing) from the video. Quite a few changes were made to improve performance.
* [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1338s)

Diffusion papers:
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
* [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

## Contents
* `train.py` - Command line utility that trains a new diffusion model on a dataset.
* `dataset.py` - Wraps a directory of image files in a PyTorch dataloader. Images can be any size or format that can be opened by PIL. All images are resized to a given dimension, converted to RGB and normalised to a range of -1 to 1.
* `modules.py` - Contains a Trainer class to handle training of the model. Contains the U-Net model and required components.
* `predict.py` - Command line utility to predict a new images from an existing `.pth` model

## Usage
### Prerequisites

* A system (preferably linux) with either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
* A GPU with at least 12GB memory if you plan to train models

### Setup

1. Clone this branch and cd to the `recognition/45802492_SimpleDiffusion/` folder
2. Setup a new conda environment. An `environment.yml` file is supplied to do this automatically.

   ```
   conda env create -f environment.yml
   conda activate diff
   
   ```
### Train a model
1. Create a folder with training images in the local directory (eg. `PatternFlow/recognition/images`). There are no requirements on image size or naming. All images within the this folder will resized and used to train the model.
2. Run the training script: `python train.py name path` which will start training. Every epoch a test image will be generated and saved to `./out` and a denoising timestep plot will be save to `./plot`.
   
3. Tensorboard is also supported and training is saved to `./runs`. You can launch tensorboard using: `tensorboard --logdir ./` to view loss metrics during training.

4. Once training has finished, the model will be saved as `name.pth` in the local directory. Additionally every epoch an `autosave.pth` file is also created.

Parameters for `train.py`

| Parameter                  | Short |          | Default                   | Description |
| ----------------           | ----- | -------- | ------------------------- | ----------- |
| _name_                     |       | required |                           | Name of model |
| _path_                     |       | required |                           | Path to dataset folder |
| _--timesteps_              |  -t   | optional | 1000                      | Number of diffusion timesteps in betas schedule|
| _--epochs_                 | -e    | optional | 100                       | Number of epochs to train for |
| _--batch_size_             | -b    | optional | 64                        | Training batch size |
| _--image_size_             | -i    | optional | 64                        | Image dimension. All images are resized to size x size |
| _--beta_schedule_          | -s    | optional | linear                    | Beta schedule type. Options: 'linear', 'cosine', 'quadratic'and 'sigmoid' |
| _--disable_images_         |       | optional |                           | Disables saving images and plots every epoch |
| _--disable_tensorboard_    |       | optional |                           | Disables tensorboard for training |

### Using an existing model
1. Run the predict script `python predict.py model`
2. A random image will be generated using the supplied model and saved

Parameters for `predict.py`
| Parameter                  | Short |          | Default                   | Description |
| ----------------           | ----- | -------- | ------------------------- | ----------- |
| _model_                    |       | required |                           | Path to `.pth` model file |
| _--output_                 |  -o   | optional | ./                        | Output path to save images|
| _--name_                   | -n    | optional | predict                   | Name prefix to use for generated images |
| _--num_images_             | -i    | optional | 1                         | Number of images to create |

Some pretrained models are supplied in the examples section below. 

## Algorithm Description
<img width="1372" alt="process" src="https://user-images.githubusercontent.com/99728921/200819420-5b0b4e26-3ca6-439d-943c-0eed254c6a5b.png">

Diffusion image generation is described in these papers: [1](https://arxiv.org/pdf/2006.11239.pdf), [2](https://arxiv.org/pdf/2105.05233.pdf). They work by describing a markov chain in which gaussain noise is sucessively added to an image for a defined number of timesteps $T$ using a variance schedule $\beta_1,...,\beta_T$.  

![Equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}q(\mathbf{x}_t|\mathbf{x}_{t-1}):=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},&space;\beta_t&space;\mathbf{I}))

This is called the forward diffusion process. The reverse diffusion process is the opposite in that given an image at a certain timestep $\mathbf{x}_t$, the denoised image is given by:

![Equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=\mathcal{N}(\mathbf{x}_{t-1};\mathbf{\mu}(\mathbf{x}_t,t),\mathbf{\Sigma}_\theta(\mathbf{x}_t,t)))

A U-Net neural network is then trained to predict the noise in an image for a given timestep. To do this, the timestep $t$ is positionally encoded using sinusoidal embeddings between the convolutional layers in the U-Net blocks. Training is performed by passing in large numbers of images from a dataset with noise added using the forward diffusion process. The U-Net is passed the noisy image and timestep as the input and the isolated noise as the target.

Once the U-Net has been trained, denoising can be performed on a random point in latent space (usually an image consisting of pure gaussian noise) using the U-Net by repeatedly subtracting the predicted noise over the entire reverse timestep range. This results in a new image that is perceptually similar to those in the training dataset.

This project uses a simplified U-Net design omitting some of the features described in the papers above. The general architecutre is:
<img width="1394" alt="unet" src="https://user-images.githubusercontent.com/99728921/200820253-d7eb02bd-bd01-4774-807f-edf451c863a1.png">

## Examples

### AKOA Knee 
Using part of the [AKOA Knee dataset](https://nda.nih.gov/oai/) consisting of 18,681 MRI images. Image size 128x128, batch size 64, 1000 Timesteps, 100 epochs. Download the pretrained [model](https://hcloudh.com/nextcloud/s/zQ4FzxGJd2aXzA8/download/AKOA2.pth).
#### Training
Epoch 0
![plot_epoch0](https://user-images.githubusercontent.com/99728921/200819735-050b1a40-d0fc-4a82-a7aa-a204effd69cf.jpeg)
Epoch 10
![plot_epoch10](https://user-images.githubusercontent.com/99728921/200819790-e74b64a3-c0c7-4181-9d8e-082c786ddefb.jpeg)
Epoch 20
![plot_epoch20](https://user-images.githubusercontent.com/99728921/200819808-bf4d4597-4c78-4163-94e2-24d38269cb7c.jpeg)
Epoch 99
![plot_epoch99](https://user-images.githubusercontent.com/99728921/200819850-38952ffc-6950-4601-bd0b-0927853887ec.jpeg)

#### Some Examples After Training
![predict0](https://user-images.githubusercontent.com/99728921/200819910-9c3baf20-3a3d-4d9a-8f49-44c93c2ccac7.jpeg)
![predict1](https://user-images.githubusercontent.com/99728921/200819962-4958c2cc-00bc-4480-bf7d-7232fe545713.jpeg)
![predict2](https://user-images.githubusercontent.com/99728921/200819973-4be54819-991c-493c-97c3-b6ef7cb83970.jpeg)
![predict3](https://user-images.githubusercontent.com/99728921/200820141-aba01039-1de4-4320-8b82-9e46e0b6a3a1.jpeg)
![predict4](https://user-images.githubusercontent.com/99728921/200819998-7f88a87c-2852-400f-9625-53b35ffea792.jpeg)
![predict5](https://user-images.githubusercontent.com/99728921/200820048-a569e7f8-0e11-4487-908a-fc84ac1d2a7c.jpeg)

### OASIS Brain
Using the [OASIS Brain](https://www.oasis-brains.org/) with 11,329 images. Image size 128x128, 1000 Timesteps, batch size 32, 100 epochs. Notice the artifacts due to the small batch size. Download the pre-trained [model](https://hcloudh.com/nextcloud/s/PpmLt5xTZLQHXHE/download/brain.pth).
#### Training
Epoch 0
![plot_epoch0](https://user-images.githubusercontent.com/99728921/200820555-6a8f8705-52a6-484a-bd5c-923dc1ba2c77.jpeg)
Epoch 10
![plot_epoch10](https://user-images.githubusercontent.com/99728921/200820567-a794ed5e-df2e-4cc2-8c58-bdc1eca44c91.jpeg)
Epoch 20
![plot_epoch20](https://user-images.githubusercontent.com/99728921/200820587-ec3a166d-c046-4fc7-a2ef-cf9773f236f7.jpeg)
Epoch 99
![plot_epoch99](https://user-images.githubusercontent.com/99728921/200820613-34b84232-a3e3-4638-be8b-e1e589ea21ab.jpeg)

#### Examples after training
![predict2](https://user-images.githubusercontent.com/99728921/200820763-4f64e55c-46b3-412d-91fb-de46f27cf097.jpeg)
![predict3](https://user-images.githubusercontent.com/99728921/200820772-26a8b5b3-9347-4a30-a58a-298794898c04.jpeg)
![predict4](https://user-images.githubusercontent.com/99728921/200820783-9c33d657-8887-44f7-9294-12c2cc4d86fc.jpeg)
![predict6](https://user-images.githubusercontent.com/99728921/200820819-aefd0c49-a90e-4dc9-a88b-e7dfbaf04221.jpeg)
![predict7](https://user-images.githubusercontent.com/99728921/200820834-9c7f0b99-9de0-4566-af0b-02916a284b5c.jpeg)
![predict8](https://user-images.githubusercontent.com/99728921/200820847-709b9103-6e96-4808-bdfb-d76c53567e75.jpeg)
![predict9](https://user-images.githubusercontent.com/99728921/200820857-accf348c-2602-44e4-9b1d-24e1d63dd566.jpeg)


### CelebA Dataset
Just for fun, the model was also trained on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset (aligned and cropped) consisting of around 200,000 images. Image size 128x128, batch size 64, 1000 Timesteps, 100 epochs. Download the pre-trained [model](https://hcloudh.com/nextcloud/s/ma6ww8GF5XEoysT/download/celebA.pth). The network does well with the faces but struggles in generating hair and backgrounds.

#### Training 
Epoch 0
![plot_epoch0](https://user-images.githubusercontent.com/99728921/200820979-d37b0b2e-b652-4b1b-b465-7c1a9fcaa7af.jpeg)
Epoch 10
![plot_epoch10](https://user-images.githubusercontent.com/99728921/200820994-e1a84d76-52bc-4f3f-8793-f5a837f2111c.jpeg)
Epoch 20
![plot_epoch20](https://user-images.githubusercontent.com/99728921/200821017-1e18b99e-8cdd-49d8-a33b-c3ad2f9e1a59.jpeg)
Epoch 99
![plot_epoch99](https://user-images.githubusercontent.com/99728921/200821040-b1f7ab29-10eb-4cd4-84f4-e2d48a45388a.jpeg)

#### Examples after training
![predict5](https://user-images.githubusercontent.com/99728921/200821204-3af9697b-4f00-4216-9779-fd30831cfb71.jpeg)
![predict6](https://user-images.githubusercontent.com/99728921/200821403-cd9b47bb-b26f-4bb7-9824-3f86323d6bc6.jpeg)
![predict7](https://user-images.githubusercontent.com/99728921/200821411-cbae264e-2236-4efc-9566-8da7779c81f6.jpeg)
![predict9](https://user-images.githubusercontent.com/99728921/200821425-fda00f0f-1a19-4f7e-a22a-e78f2767543a.jpeg)
![predict11](https://user-images.githubusercontent.com/99728921/200821444-c2815252-6e5a-490c-b4af-1adeba572d3a.jpeg)
![predict13](https://user-images.githubusercontent.com/99728921/200821475-b05a72ef-a30f-46ad-b058-1dd5146c48f0.jpeg)
![predict12](https://user-images.githubusercontent.com/99728921/200821515-49690075-31b5-42b2-84fa-10d3d6e0a246.jpeg)

