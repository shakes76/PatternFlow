# Diffusion Image Generation
Diffusion based image generation using pytorch.
![](https://hcloudh.com/nextcloud/s/YnMBoAK6atDYztj/download/plot_epoch98.jpeg)
#### References
Huge thanks to these videos for helping my understanding:
* [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=912s)
    * The code base was built using heavy references from this [colab notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing) that accompanies the first video.
* [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1338s)  


## Usage
Two command line modules are provided: `train.py` and  `predict.py` that can be used to train models and predict new images from existing models respectively.

### Prerequisites
  * A system (preferably linux) with either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
  * A beefy GPU with at least 12GB memory if you plan to train models

### Steps
1. Clone this branch and cd to the `recognition` folder
2. Setup a new conda environment. An  `environment.yml` file is supplied to do this automatically.
    ```bash
    conda env create -f environment.yml
    conda activate diff
    ```
    Grab a cup of coffee, this may take a while.

#### Train a model
1. Create a folder with training images in the local directory (ie. `PatternFlow/recognition'). There are no requirements on image size or naming. All images within the this folder will resized and used to train the model.
2. Run the training script:
    ```
    python train.py some_name path_to_dataset
    ```  
    This will start training. Every epoch a test image will be generated and saved to `./out` and a denoising timestep plot will be save to `./plot`. 
3. Run the training with optional command line arguments. You can set the image size, number of epochs and other additional settings. To view additional arguments run:
    ```
    python train.py --help
    ```
4. Tensorboard is also supported and training is saved to `./runs`. You can launch tensorboard using:
    ```
    tensorboard --logdir ./
    ```
5. Once training has finished, the model will be saved as `some_name.pth` in the local directory. Additionally every epoch an `autosave.pth` file is also created.
    
#### Using an existing model
