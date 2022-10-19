# Simple Diffusion Image Generation

Simple diffusion based image generation using pytorch.
![](https://hcloudh.com/nextcloud/s/TpPW8Z3EzCc3mP6/download/plot_epoch94.jpeg)
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
![](https://hcloudh.com/nextcloud/s/kRoAMTSQkJ4weZf/download/process.png)
Diffusion image generation is described in these papers: [1](https://arxiv.org/pdf/2006.11239.pdf), [2](https://arxiv.org/pdf/2105.05233.pdf). They work by describing a markov chain in which gaussain noise is sucessively added to an image for a defined number of timesteps $T$ using a variance schedule $\beta_1,...,\beta_T$.  

![Equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}q(\mathbf{x}_t|\mathbf{x}_{t-1}):=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},&space;\beta_t&space;\mathbf{I}))

This is called the forward diffusion process. The reverse diffusion process is the opposite in that given an image at a certain timestep $\mathbf{x}_t$, the denoised image is given by:

![Equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=\mathcal{N}(\mathbf{x}_{t-1};\mathbf{\mu}(\mathbf{x}_t,t),\mathbf{\Sigma}_\theta(\mathbf{x}_t,t)))

A U-Net neural network is then trained to predict the noise in an image for a given timestep. To do this, the timestep $t$ is positionally encoded using sinusoidal embeddings between the convolutional layers in the U-Net blocks. Training is performed by passing in large numbers of images from a dataset with noise added using the forward diffusion process. The U-Net is passed the noisy image and timestep as the input and the isolated noise as the target.

Once the U-Net has been trained, denoising can be performed on a random point in latent space (usually an image consisting of pure gaussian noise) using the U-Net by repeatedly subtracting the predicted noise over the entire reverse timestep range. This results in a new image that is perceptually similar to those in the training dataset.

This project uses a simplified U-Net design omitting some of the features described in the papers above. The general architecutre is:
![](https://hcloudh.com/nextcloud/s/atkz7cBTeJEDAbW/download/unet.png)

## Examples

### AKOA Knee 
Using part of the [AKOA Knee dataset](https://nda.nih.gov/oai/) consisting of 18,681 MRI images. Image size 128x128, batch size 64, 1000 Timesteps, 100 epochs. Download the pretrained [model](https://hcloudh.com/nextcloud/s/zQ4FzxGJd2aXzA8/download/AKOA2.pth).
#### Training
Epoch 0
![](https://hcloudh.com/nextcloud/s/LdiWLFwsMtqGCPy/download/plot_epoch0.jpeg)
Epoch 10
![](https://hcloudh.com/nextcloud/s/Ho8apXrybbamXrS/download/plot_epoch10.jpeg)
Epoch 20
![](https://hcloudh.com/nextcloud/s/L6SfgoKzpWdCWam/download/plot_epoch20.jpeg)
Epoch 99
![](https://hcloudh.com/nextcloud/s/CZraXEaTGactfaz/download/plot_epoch99.jpeg)

#### Some Examples After Training
![](https://hcloudh.com/nextcloud/s/LEpJzmtMFHY2CyS/download/predict0.jpeg)![](https://hcloudh.com/nextcloud/s/EJCRb3wj43D75qN/download/predict1.jpeg)![](https://hcloudh.com/nextcloud/s/NT7RY7b7sLeXGSF/download/predict2.jpeg)![](https://hcloudh.com/nextcloud/s/KtfFcEYSjnyxiwz/download/predict3.jpeg)![](https://hcloudh.com/nextcloud/s/MzTR56qysE3N5pT/download/predict4.jpeg)![](https://hcloudh.com/nextcloud/s/WtKYfsEdo8nYq5q/download/predict5.jpeg)![](https://hcloudh.com/nextcloud/s/kTsjDgQSdfAibFY/download/predict6.jpeg)

### OASIS Brain
Using the [OASIS Brain](https://www.oasis-brains.org/) with 11,329 images. Image size 128x128, 1000 Timesteps, batch size 32, 100 epochs. Notice the artifacts due to the small batch size. Download the pre-trained [model](https://hcloudh.com/nextcloud/s/PpmLt5xTZLQHXHE/download/brain.pth).
#### Training
Epoch 0
![](https://hcloudh.com/nextcloud/s/rq2TPtAp3Aibj5x/download/plot_epoch0.jpeg)
Epoch 10
![](https://hcloudh.com/nextcloud/s/r3LLCparqbo77Xw/download/plot_epoch10.jpeg)
Epoch 20
![](https://hcloudh.com/nextcloud/s/PGaqHXzowryKtkB/download/plot_epoch20.jpeg)
Epoch 99
![](https://hcloudh.com/nextcloud/s/rNaFJptFKMTnz4X/download/plot_epoch99.jpeg)

#### Examples after training
![](https://hcloudh.com/nextcloud/s/QDdoBxFGF5RNLXg/download/predict2.jpeg)![](https://hcloudh.com/nextcloud/s/qZiA5zPDAxSAXgx/download/predict3.jpeg)![](https://hcloudh.com/nextcloud/s/84ftrCnnXJmgjGp/download/predict4.jpeg)![](https://hcloudh.com/nextcloud/s/EEodLTB24xiw6x2/download/predict6.jpeg)![](https://hcloudh.com/nextcloud/s/WjWceaEJ5BsbtJb/download/predict7.jpeg)![](https://hcloudh.com/nextcloud/s/4dkMk97t4KgFRRs/download/predict8.jpeg)![](https://hcloudh.com/nextcloud/s/S67ByeBPYtbq52G/download/predict9.jpeg)

### CelebA Dataset
Just for fun, the model was also trained on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset (aligned and cropped) consisting of around 200,000 images. Image size 128x128, batch size 64, 1000 Timesteps, 100 epochs. Download the pre-trained [model](https://hcloudh.com/nextcloud/s/ma6ww8GF5XEoysT/download/celebA.pth). The network does well with the faces but struggles in generating hair and backgrounds.

#### Training 
Epoch 0
![](https://hcloudh.com/nextcloud/s/z3z6qcX2bz3dZGy/download/plot_epoch0.jpeg)
Epoch 10
![](https://hcloudh.com/nextcloud/s/weKD2Abeg57KD2w/download/plot_epoch10.jpeg)
Epoch 20
![](https://hcloudh.com/nextcloud/s/jSW944KfLSXXX9s/download/plot_epoch20.jpeg)
Epoch 99
![](https://hcloudh.com/nextcloud/s/toiR4mKsXemxaxM/download/plot_epoch99.jpeg)
#### Examples after training
![](https://hcloudh.com/nextcloud/s/xiEtPoB73REYZZk/download/predict5.jpeg)![](https://hcloudh.com/nextcloud/s/ggdw7wtLEEEWyoa/download/predict6.jpeg)![](https://hcloudh.com/nextcloud/s/jXaebSRSTpjC9Yq/download/predict7.jpeg)![](https://hcloudh.com/nextcloud/s/AiLLPETBajJbKzd/download/predict9.jpeg)
![](https://hcloudh.com/nextcloud/s/DYZCgkToCCMAg6o/download/predict11.jpeg)![](https://hcloudh.com/nextcloud/s/XC3a8EdwFtYF8gd/download/predict13.jpeg)![](https://hcloudh.com/nextcloud/s/BZa4PCZTPATniTs/download/predict12.jpeg)
