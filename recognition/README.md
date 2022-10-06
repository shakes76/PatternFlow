# Diffusion Image Generation

Simple diffusion based image generation using pytorch.
![](https://hcloudh.com/nextcloud/s/YnMBoAK6atDYztj/download/plot_epoch98.jpeg)
#### References

Huge thanks to these videos for helping my understanding:

* [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=912s)
  * This repo was built largely referencing code in this [colab notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing) from the video. Quite a few changes were made to improve performance.
* [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1338s)

## Usage

Two command line modules are provided: `train.py` and `predict.py` that can be used to train models and predict new images from existing models respectively. A pre-trained model that was trained on a portion of the [OAI Knee](https://nda.nih.gov/oai/) dataset is also supplied.

### Prerequisites

* A system (preferably linux) with either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
* A GPU with at least 12GB memory if you plan to train models

### Setup

1. Clone this branch and cd to the `recognition` folder
2. Setup a new conda environment. An `environment.yml` file is supplied to do this automatically.

   ```
   conda env create -f environment.yml
   conda activate diff
   
   ```
### Train a model
1. Create a folder with training images in the local directory (eg. `PatternFlow/recognition/images`). There are no requirements on image size or naming. All images within the this folder will resized and used to train the model.
2. Run the training script: `python train.py name path` which will start training. Every epoch a test image will be generated and saved to `./out` and a denoising timestep plot will be save to `./plot`.
   
3. Tensorboard is also supported and training is saved to `./runs`. You can launch tensorboard using: `tensorboard --logdir ./`

4. Once training has finished, the model will be saved as `name.pth` in the local directory. Additionally every epoch an `autosave.pth` file is also created.

Parameters for `train.py`

| Parameter                  | Short |          | Default                   | Description |
| ----------------           | ----- | -------- | ------------------------- | ----------- |
| _name_                     |       | required |                           | Name of model |
| _path_                     |       | required |                           | Path to dataset folder |
| _--timesteps_              |  -t   | optional | 300                       | Number of diffusion timesteps in betas schedule|
| _--epochs_                 | -e    | optional | 100                       | Number of epochs to train for |
| _--batch_size_             | -b    | optional | 128                       | Training batch size |
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
| _model_                    |       | required |                           | Path to model file |
| _--output_                 |  -o   | optional | ./                        | Output path to save images|
| _--name_                   | -     | optional | predict                   | Name prefix to use for generated images |
| _--seed_                   | -s    | optional | None                      | Passing in an integer seed will yield reproducable images|

## Theory
