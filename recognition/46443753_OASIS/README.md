# Generative Model for OASIS Brain Dataset using StyleGAN

Author: Danny Wang

## Problem Statement & OASIS Brian Dataset Overview

For this project, a generative adversarial network (GAN) following the StyleGAN architecture proposed in the paper by T Karras et al.[1],  has been implemented using the OASIS Brain dataset. With an attempt to generate "reasonably clear images" of the brain MRI.

The OASIS Brain dataset consists of a total of 11328 brain MRI images (combining train, test and validation samples), with a size of 256 by 256 pixels each.

Selected example:

<p align="center">
  <img src="examples/case_441_slice_0.nii.png" />
 </br>
 <em>Figure 1: Example sample of OASIS Brian Data</em></p>

## StyleGAN Architecture & Description

Based on the paper by T Karras et al.[1], the main improvement of StyleGAN from traditional GAN models comes from the introduction of the mapping network. In the essense, StyleGAN is a continuation of the progressive GAN model. In which the discriminator model remains similar to the progressive GAN model, where it continues to downsample the input data to output a boolean value for determining the realistic of the input. Whilst the generator also exhibit the "progressiveness", as indicated in the continue upsampling of data in the synthesis network as shown in figure 2. Through introducing the mapping network, it removes the need to directly feed in the latent space to the generator, and we can have instead a constant as the initial input to the synthesis network. In this way, the mapping network will help to disentangle latent space vector, and maps into a cloud of intermediate latent vector w to have more control of the . From here, we can see from the figure, after some affine transform A of the latent vector w, we can apply it to the adaptive instance normalisation (AdaIN) layer in the model to gain much more control of the various "styles" of the generated image. In addition, a along with the AdaIN layer, before feeding any output to the AdaIN layer, some uncorrelated gaussian noise B scaled by a learnt per-feature scalling factor was added to the convolution output. This way, it will allow finer control in the stochastic details of the generated image. Hence, we can see that the generator model will take three inputs, namely the latent space Z, a constant vector for the synthesis network g and the randomly generated noise inputs.

<p align="center">
  <img src="examples/stylgan.png" alt="stylegan" />
 </br>
 <em>Figure 2: StyleGAN Architecture</em></p>

Moreover, based on the discussed architecture, due to computation limits, some slight modification was made to parameters of the architecture, which differes to the ones mentioned in the paper. This includes a filter size of 256 for the fully connected layers in the mapping network, a channel size of 128 in the constant input vector as compared to 512, and a latent dimension of 128. A detailed model summary for the generator and discriminator was provided in the Appendix.

## Model Dependencies & Preprocessing

Library dependencies are as follow:

| Library    | Version |
| ---------- | :-----: |
| Python     |  3.8.5  |
| Tensorflow |  2.7.0  |
| Tqdm       | 4.64.0 |
| Matplotlib |  3.3.2  |
| Numpy      | 1.19.2 |

In this generative project, the OASIS dataset was utilised as the training set, and randomly generated gaussian normal noise was utilised as test input for the StyleGAN model. The training images was loaded in as grayscale values [0, 255], and was normalised to [0, 1] for training.

Whilst there are many factors that may impact on the performance of an generative adversarial network, for reproducibility purposes, the basic model parameters used to train this model includes:

| Parameter                                  | Value |
| ------------------------------------------ | :----: |
| Epochs                                     |  120  |
| Batch Size                                 |   12   |
| Generator Adam optimizer learning rate    |  2e-7  |
| Discriminator Adam optimizer learning rate | 1.5e-7 |
| Optimizer beta 1                           |  0.5  |
| Optimizer beta 2                           |  0.99  |
| Image Size                                 |  256  |
| Latent Dimension                           |  128  |

## Usage

In this folder, there are 4 main python scripts.

- train.py: the driver script that is required to be run for training and saving model.
- dataset.py: containing the data loader and preprocessing function.
- modules.py: contains the generator and discrimnator's model components.
- predict.py: shows an exmaple usage of the trained model by generating specified numbers of samples from the saved model.

Ensure the dependencies are met to ensure the successful running of the train.py script. The following global parameters are required to be confirmed prior of running the script.

train.py:

- `PIC_DIR:` List of directories to the stored data.
- `EPOCHS` : Number of epochs to train the model, default to 120 epochs.
- `LATENT_DIM `: Latent dimension value, default to 128.

predict.py:

- `NUM_SAMPLES`: Number of samples to be generated, default to 9.
- `PLOT_SHAPE`: Shape of the output plot, default to 3 by 3.
- `CHECKPOINT_DIRECTORY`: Directory to the saved model checkpoint.

## Model Performance

The following figures illustrates the results during training:

<table class="image-grid">
    <tr>
        <td>
            <img src="./examples/generated_plot_e040.png" alt="Epoch 40"/>
        </td>
        <td>
            <img src="./examples/generated_plot_e080.png" alt="Epoch 80"/>
        </td>
    </tr>
   <tr>
        <td align="center">
            Epoch 40
        </td>
        <td align="center">
            Epoch 80
        </td>
    </tr>
</table>

<p align="center">
     <img src="./examples/generated_plot_e120.png" alt="Epoch 120" width="450" 
     height="350"/>
    <p align="center">Epoch 120</p>
</p>

<p align="center">
    <em>Fig 3: Trianing progress </em>
</p>

After training for 120 epochs with batch size of 14 and 13000 iterations in each epoch, the training loss for the generator and discriminator are given below.

<p align="center">
     <img src="./examples/model_loss.png" alt="Model Loss" width="400" 
     height="300"/>
<br/>
    <em align="center"> Figure 4: Training Loss</em></p>
</p>

Using the trained model ,we can generate some samples of the brain MRI images given some randomly generated inputs. In which, it is clear that whilst the result doesn't provide the same level of details as the orginal OASIS dataset image. The general shape and clarity of the image could be seen, and potentially interpreted as a brian MRI. Thus, it is believed that with more training and parameter tuning, the generated output could be greatly improved.

<table class="image-grid">
    <tr>
        <td>
            <img src="./examples/Sample_output.png" alt="Sample Output" width="400"/>
        </td>
        <td>
            <img src="./examples/Sample_output2.png" alt="Sample Output" width="400"/>
        </td>
    </tr>
</table>
<p align="center">
    <em align="center"> Figure 5: Generated Samples</em></p>

---

## References:

1. [StyleGAN Paper](https://arxiv.org/pdf/1812.04948.pdf)
2. [OASIS Brain](https://www.oasis-brains.org/)
3. [StyleGAN Overview](https://www.analyticsvidhya.com/blog/2021/05/stylegan-explained-in-less-than-five-minutes/)

## Appendix

A. Generator model summary

<p align="center">
     <img src="./examples/g_model.png" alt="Generator model"/>
<br/>
    <em align="center"> Figure 6: Generator model</em></p>
</p>

B. Discriminator model

<p align="center">
     <img src="./examples/d_model.png" alt="discriminator model"/>
<br/>
    <em align="center"> Figure 7: Discriminator model</em></p>
</p>
