# Pattern Recognition: StyleGAN for the OASIS brain dataset

## Author

Keith Dao (45798906)

## Problem Overview

GANs allow for the generation of synthetic but real looking data. The most common use case of GANs are images, which is also being tackled here. A GAN model following the StyleGAN architecture must be used to generate real looking MRIs of a brain. The OASIS brain dataset had already been preprocessed and is ready to be used as training data for the GAN. Ideally, the GAN would be able to generate MRIs with features similar to those in the OASIS dataset.

## Description of StyleGAN

### What is a GAN?

A GAN or Generative Adversarial Model consists of two neural networks, called the discriminator and the generator. The job of the discriminator is to guess whether or not the image it is given is real or fake, while its the generator's job to fool the discriminator into believing the generated image is real. The adversarial aspect of GANs comes from the discriminator and generator constantly trying to beat one another. Additionally, the real images are never seen by the generator, but learns from the discriminator's incorrect guesses on the fake images.

### What are some GANs?

Some well-known GANs are DCGGAN, ProGAN and StyleGAN. DCGAN (Deep Convolutional GAN) is simply a GAN that uses a deep convolutional neural network for both the discriminator and generator. Both the ProGAN and StyleGAN build upon this architecture to generate more realistic looking images. ProGAN (Progressively Growing GAN) builds upon the DCGAN by progressively growing the resolution the GAN is trained on, which allowed the network to capture broader details first and slowly add more details as the resolution increases.

### How is StyleGAN different?

StyleGAN builds upon ProGAN by introducing a mapping network for the latent vector, which feeds into the Adaptive Instance Normalisation layers throughout the generator, and the addition of noise throughout the generator. The introduction of a mapping network removes the need to directly feed the latent code to the generator, rather a constant value is used instead as the input of the generator.

<p align="center">
    <img src="resources/stylegan_architecture.png" alt="StyleGAN architecture" />
    <p> General architecture of GAN (left) and StyleGAN (right). Obtained from https://arxiv.org/abs/1812.04948  </p>
</p>

As briefly described above, the latent vector is fed into the mapping network rather than the synthesis network. From the output of the mapping network, the learned affine transform, represented as "A", is obtained. This affine transform is used to generate the weight and bias terms, <img src="https://render.githubusercontent.com/render/math?math=\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> respectively, for Adaptive Instance Normalisation (AdaIN). The equation of AdaIN is given by:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{AdaIN}(x_i, \gamma, \beta) = \gamma \frac{x_i-\mu(x_i)}{\sigma(x_i)} + \beta">
</p>

The use of AdaIN replaces the need for any other normalisation layers as it provides control over the style.

The other input receives single-channel images of uncorrelated Gaussian noise with the same resolution as the convolution results it is being added to. But before the noise is added to the convolution result, the noise weighted via learned per-feature scaling factors. This allows slight variations in the noise image to adjust minor details of the image without changing the overall appearance of the image.

## Dependencies

Python version: 3.9.7

| Library    | Version |
| ---------- | ------- |
| TensorFlow | 2.6.0   |
| Matplotlib | 3.4.2   |
| Tqdm       | 4.62.2  |

The versions listed above are the versions used to test/run the scripts would be the most stable.

`TensorFlow` was used to construct and train the GAN and load the training data.  
`Matplotlib` was used to visualise the model losses and the generator's images.  
`Tqdm` was used to provide visualisation of the training epoch's progress.

## Methodology

### Data loading

The images were loaded using the `TensorFlow Keras` API, which allowed the images to be directly imported into a TensorFlow dataset in the greyscale format (1 colour channel as opposed to RGB with 3 colour channels) via the `tf.keras.preprocessing.image_dataset_from_directory` method. These images are then normalised from [0, 255] to [0, 1].

### Training, validation, test split

Although it is typical to separate data into training, validation and test dataset when training neural networks, it does not provide much value when it comes to GANs as a whole. The data split may be useful if the performance of the discriminator is the most important. However, for this task, the images generator by the generator is more important than the capabilities of the discriminator. Thus, it was decided that all the data would be used for training and no splitting would be performed.

### Data augmentation

To increase the range of possible MRI brains generated, the training data is randomly flipped across the horizontal axes. This retains the position of where the front and back of the path are, while flipping the left and right hemispheres of the brain. This essentially doubles the training domain, as it is unlikely a perfectly matching brain is part of the dataset. This random flipping is performed after the dataset has been exhausted, which can aid in preventing the discriminator from overfitting and reduce the training time of each epoch.

### Model construction

The general architecture was discussed in [How is StyleGAN different?](#how-is-stylegan-different). `Keras` had a majority of the layers to implement the model with the exception of AdaIN. The AdaIN layer was created as a subclass of `tf.keras.layers.Layer` and required `build` and `call` to be implemented to work as the paper intended. The remaining layers were imported from `tf.keras.layers`. The network was built as a [functional model](https://keras.io/guides/functional_api/), which allows for greater control over the inputs of a layer when compared to the sequential model. In order to reduce the amount of repeated layers, the methods `gen_block` and `disc_block` were created to create generator and discriminator blocks respectively at various resolutions. With these blocks and several other layers, the generator and discriminator are built in the `get_generator` and `get_discriminator` functions respectively in [model.py](model.py).

### Visualisation

#### Training

The progress bar displayed during training is achieved via the use of `tqdm`.

### Graphs and Images

The graphs and images are display using `matplotlib.pyplot`. The graphs are achieved via `plt.plot` while the images are displayed using `plt.subplot` to create a grid and `plt.imshow` to convert the tenors into an image and be displayed in a subplot of the whole figure.

## Results

### Recommended Environment

| Training environment | Recommendations              | Reasoning                                                                                                                                                                                                                                                                      |
| -------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 32GB of RAM          | Minimum 16GB of RAM          | Training uses 10GB+ of RAM.                                                                                                                                                                                                                                                    |
| Nvdia RTX2070 8GB    | Nvidia GPU with 8GB+ of VRAM | Training uses 7GB+ of VRAM. More VRAM would also allow for higher resolution images and batch sizes. A GPU is MANDATORY to be able to train in a timely manner. Although an AMD GPU can be used, additional libraries need to be instead and has not been tested to be stable. |

### Training Parameters

<table>
    <tr> 
        <td>
            <table>
                <thead>
                    <tr>
                        <th>Hyperparameter</th>
                        <th>Generator</th>
                        <th>Discriminator</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Optimizer</td>
                        <td colspan=2 align="center">Adam</td>
                    </tr>
                    <tr>
                        <td>Learning Rate</td>
                        <td align="center">2e-7</td>
                        <td align="center">2.5e-8</td>
                    </tr>
                    <tr>
                        <td>Beta 1</td>
                        <td colspan=2 align="center">0.5</td>
                    </tr>
                    <tr>
                        <td>Beta 2</td>
                        <td colspan=2 align="center">0.999</td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Image Size</td>
                        <td align="center">256</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td align="center">32</td>
                    </tr>
                    <tr>
                        <td>Number of filters</td>
                        <td align="center"> 512</td>
                    </tr>
                    <tr>
                        <td>Latent dimension</td>
                        <td align="center">512</td>
                    </tr>
                    <tr>
                        <td>Kernel size</td>
                        <td align="center">3</td>
                    </tr>
                    <tr>
                        <td>Total Epochs</td>
                        <td align="center">200</td>
                    </tr>
                </tbody>
            </table>
        </td>
    </tr>
</table>

After many trials, the hyperparameters above would be able to produce the results below. Increasing the learning rate to around 1e-5 would most likely cause the generator to suffer from mode collapse after 50 epochs. A learning rate ratio of 8:1 (generator : discriminator) seemed to provide enough detail, lowering this ratio appeared to cause finer details to be smeared.

### Training Results

The following are some samples of the results achieved when training the model on the parameters listed above.

<table class="image-grid">
    <tr>
        <td align="center">
            Epoch 154
        </td>
        <td align="center">
            Epoch 158
        </td>
    </tr>
    <tr>
        <td>
            <img src="resources/epoch_154.png" alt="Epoch 154"/>
        </td>
        <td>
            <img src="resources/epoch_158.png" alt="Epoch 158"/>
        </td>
    </tr>
    <tr>
        <td align="center">
            Epoch 159
        </td>
        <td align="center">
            Epoch 161
        </td>
    </tr>
    <tr>
        <td>
            <img src="resources/epoch_159.png" alt="Epoch 159"/>
        </td>
        <td>
            <img src="resources/epoch_161.png" alt="Epoch 161"/>
        </td>
    </tr>
    <tr>
        <td align="center">
            Epoch 162
        </td>
        <td align="center">
            Epoch 167
        </td>
    </tr>
    <tr>
        <td>
            <img src="resources/epoch_162.png" alt="Epoch 162"/>
        </td>
        <td>
            <img src="resources/epoch_167.png" alt="Epoch 167"/>
        </td>
    </tr>
    <tr>
        <td align="center">
            Epoch 195
        </td>
        <td align="center">
            Epoch 200
        </td>
    </tr>
    <tr>
        <td>
            <img src="resources/epoch_195.png" alt="Epoch 195"/>
        </td>
        <td>
            <img src="resources/epoch_200.png" alt="Epoch 200"/>
        </td>
    </tr>
</table>

<p align="center">
    <img src="./resources/final_evolution.gif" alt="Training sample evolution"/>
    <p align="center">Training Samples</p>
</p>

Although samples are not perfect, the shape and some details of the brain MRIs can be seen. Some higher quality images could possibly be generated by lowering the learning rate further then training for more epochs or adding more training data.

<p align="center">
    <img src="resources/final_loss.png" alt="GAN loss graph" />
    <p align="center">GAN Loss</p>
</p>

Although it may appear the loss has converged, it is quite evident that the images generated by the generator are definitely improving after the 20th epoch.

## Repository Overview

`resources` contains the images used in this README.  
`model.py` contains the helper functions required to generate and train the StyleGAN models.  
`util.py` contains the helper functions required to load, augment and visualise the training images. Also includes helper functions to generate and visualise the loss history plots and save any figures.  
`driver.py` uses the functions from `model.py` and `util.py` to train the network and show the results of the training. For instructions on how to configure `driver.py`, see [Usage](#usage).

## Usage

Before running `driver.py`, the [dependencies](#dependencies) must be met. The global variables must be configured before running to prevent undesired outcomes, as the filepaths used are most likely configured appropriately. The variables are explained below.

**NOTE:** All file paths must end with a file separator. These paths can be a relative or absolute path.  
For UNIX based systems, the file separator is "/" i.e. `"dir/sub_dir/file`.  
For Windows systems, the file separator is "\\" but would need to be escaped unless raw strings are used i.e. `"dir\\sub_dir\\file"` or `r"dir\sub_dir\file"`

Training variables:

- `TRAIN`: A boolean of whether or not the model should be trained.
- `IMAGE_PATHS`: A list of all the training data directories.
- `EPOCHS`: The number of epochs to train when the script is ran.
- `TOTAL_PREVIOUS_EPOCHS`: The number of epochs that has been previously ran. If the model does not load any weights, this is automatically set to 0.
- `MODEL_NAME`: A string to name the model. Used when creating output directories and files.

Model weight variables:

- `LOAD_WEIGHT`: A boolean of whether or not weights should be loaded.
- `GENERATOR_WEIGHT_PATH`: Path to the generator weight.
- `DISCRIMINATOR_WEIGHT_PATH`: Path to the discriminator weight.
- `SAVE_WEIGHTS`: A boolean of whether or not the weights should be saved.
- `WEIGHT_SAVING_INTERVAL`: An integer signifying how often the weights should be saved. Setting this to `EPOCHS` causes the weights to only be saved when training has completed.
- `WEIGHT_PATH`: Path to save the generator and discriminator weights.

Image sampling variables:

- `SHOW_FINAL_SAMPLE_IMAGES`: A boolean of whether or not the final epoch's results should be displayed in a new window.
- `SAVE_SAMPLE_IMAGES`: A boolean of whether or not the save the generator's results after training `IMAGE_SAVING_INTERVAL` epochs.
- `IMAGE_SAVING_INTERVAL`: An integer signifying how often the generator results should be saved.
- `SAMPLE_IMAGES_PATH`: Path to save the generator results.

Model loss visualisation variables:

- `VISUALISE_LOSS`: A boolean of whether or not the loss history should be displayed in a new window.
- `SAVE_LOSS`: A boolean of whether or not the loss history should be saved.
- `LOSS_PATH`: Path to save the loss history to.

## References

- StyleGAN paper, Available at: https://arxiv.org/abs/1812.04948
- GAN Training, Available at: https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
- StyleGAN Keras implementation, Available at: https://github.com/manicman1999/StyleGAN-Keras
