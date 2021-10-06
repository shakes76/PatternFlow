<h1>Segment the ISICs data set with the Improved UNet</h1>
This repository contains the tensorflow implementation of the Improved UNet.

<h1>Instructions</h1>
Modules:
import matplotlib.pyplot: conda install matplotlib
import tensorflow: conda install tensorflow

Information about ISICs:
The International Skin Imaging Collaboration (ISIC) aims to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermoscopic images of skin lesions.

<h1>Resources:</h1>
The origin image sets can be downloaded via: https://challenge.isic-archive.com/data
The images contain the training image and the ground truth images

<h1>Requirements:</h1>
Python 3.7
TensorFlow 2.1.0
matplotlib 3.3.1
ISICs 2018 Challenge dataset (The download link is provided in the Resources)

<h1>Explanation:</h1>
The style based GAN model is an extension to the GAN architecture that proposes large changes to the generator model, including the use of a mapping network to map points in latent space to an intermediate latent space, the use of the intermediate latent space to control style at each point in the generator model, and the introduction to noise as a source of variation at each point in the generator model.
