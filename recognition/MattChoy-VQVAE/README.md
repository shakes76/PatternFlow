# OASIS Generative VQ-VAE
By Matthew Choy
> Creation of a generative model of the ADNI dataset using the VQVAE model ([paper](arxiv.org/abs/1711.00937))

# Files of Interest
What files to look at, and what purpose they serve.
- [modules.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/modules.py) Contains module definitions for the VQVAE (& Encoder, Decoder) as well as PixelCNN
- [train.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/train.py) Uses the modules defined in `modules.py` to train and save the VQVAE and PixelCNN models.
- [dataset.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/dataset.py) Loads and normalises dataset.
- [constants.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/constants.py) Defines constants and hyperparameters to ensure that this model can be adapted to other datasets more easily.
- [predict.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/predict.py) Loads the trained VQVAE and PixelCNN models located in `./vqvae` and `./pixel_cnn` respectively to generate OASIS-like brain scan images.
- [util.py](https://github.com/MattPChoy/PatternFlow/blob/topic-recognition/recognition/MattChoy-VQVAE/util.py) Defines helper methods for visualisation, data processing etc.

## Table of Contents
- [Algorithm Description](#description-of-algorithm)
- [How Does it Work?](#how-does-it-work)
- [Inputs, Outputs and Algorithm Performance](#inputs-outputs-and-algorithm-performance)
- [Implementation Details](#implementation-details)
  - [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Pre-Processing](#data-pre-processing)
- [Training, Validation and Testing Splits](#training-validation-and-testing-splits)

### Description of Algorithm
- What is this algorithm that is implemented, and what problem does it solve?
- Approximately a paragrahm
- Screenshots and conceptual explanation from the research paper.

### How does it work?
- Approximately a paragraph
- Include a visualisation

## Inputs, Outputs and Algorithm Performance
Images fed into the model are of dimensionality $(\text{batch\_size}, 256, 256, 1)$ and are taken from the OASIS brain dataset. Some sample images include those shown below:
<p align="center">
  <img style="width:60vw" src="./images/sample_inputs.png"/>
</p>

## Implementation Details
### Dependencies
- Add a list of project dependencies, here, via `conda list`.

### Dataset
- The dataset used for this task was downloaded from BlackBoard - [COMP3710/"Course Help/Resources"/ADNI MRI Dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)
### Data Pre-Processing
- The data was unzipped from the above .zip file, and moved into the project's root folder (PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/)
### Training, Validation and Testing Splits
