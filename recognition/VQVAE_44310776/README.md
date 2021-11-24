# VQVAE2 and PixelCNN
[VQVAE2 [1]](https://arxiv.org/abs/1906.00446) is an autoencoder network which can be used for compression and denoising of images. It can also be used as a generative model when paired with an autoregressive network such as PixelCNN. The model introduces a hierarchical architecture (Figure 2) to the original [VQVAE [2]](https://arxiv.org/abs/1711.00937) (Figure 1) which significantly improves both reconstruction and sampling performance. The discrete latent spaces learned by the network can generate high-quality images with more diversity than most GANs.

| ![vqvae.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0f22d1a8-a53a-415e-91da-dba0ddc2544c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211122%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211122T222011Z&X-Amz-Expires=86400&X-Amz-Signature=4aecb5feb14a8a37f7c00905419a0dac5e3873c2931189b4ed73c02753f4c5ae&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject) | 
|:--:| 
| Figure 1. Original VQVAE architecture (left) and codebook learning mechanism (right) [2]. |

| ![vqvae2.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5840479c-2447-462b-98bc-f2278577d018/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211122%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211122T222213Z&X-Amz-Expires=86400&X-Amz-Signature=b7a116289313091c28291af818d883c75bc95a7a31926388383b467707386b36&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject) | 
|:--:| 
| Figure 2. VQVAE2 architecture [1]. |

Like VQVAE, VQVAE2 uses residual convolutional encoder and decoder networks to compress images. The latent space in these models is discrete, unlike traditional VAEs, and is constructed through quantization of the encoder output; the vectors are "snapped" to the closest vector in the embedding space (codebook). The vectors in the codebook are learned through the training process as illustrated in Figure 1 (right). This implementation uses Exponential Moving Averages for the learning process. Some sample reconstructions from this implementation trained on the OASIS Brain MRI dataset are shown in Figure 3.

| ![reconstructions.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5de762c7-1f12-45cd-a7d4-54f33ca247ef/oasis_reconstructions.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211122%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211122T222146Z&X-Amz-Expires=86400&X-Amz-Signature=b7402bcb208126abe0b9a3d37061c0ca9ccbadcadb55a6482e8f962d58b6c5ed&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22oasis_reconstructions.png%22&x-id=GetObject) | 
|:--:| 
| Figure 3. Reconstruction samples from the OASIS Brain MRI dataset. |

Once the model is trained, the discrete latent space can be sampled from to generate new images as illustrated in Figure 4. This is done using an autoregressive model. This implementation uses the same network as the authors - [PixelCNN [3]](https://arxiv.org/abs/1606.05328) - but using an RNN or a Transformer is possible too. This model is used to generate latent representations, rather than images themselves, which are decoded to produce an image. This approach is much faster than gererating images pixel-by-pixel.

| ![generation.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b2693f7d-b634-408c-8395-04671820167f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211122%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211122T222127Z&X-Amz-Expires=86400&X-Amz-Signature=c3259472ddad4ed5b5822784e5c6065a11d195db3745d952018efd1dd56bac92&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject) | 
|:--:| 
| Figure 4. Sampling process for VQVAE2 using PixelCNN. |

Two PixelCNN networks are used in this instance; one samples from the top-level latent space, and the other from the bottom-level. The top-level PixelCNN includes additional self-attention layers for improved global context (as done by the authors). The bottom-level network does not include these due to computation constraints, but is conditioned on the top-level network as in Figure 3. A few generated samples from the OASIS Brain MRI dataset are shown in Figure 5.

| ![samples.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/254b185b-ebb0-45cd-94f5-3284fd50896a/sampled_brains.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20211122%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211122T222158Z&X-Amz-Expires=86400&X-Amz-Signature=685767723c3b714737b221ba0aa4c0ca44b47d23397ad3af2e97ca9ca3e4f064&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22sampled_brains.png%22&x-id=GetObject) | 
|:--:| 
| Figure 5. Random samples from the OASIS Brain MRI dataset. |

## Training Process
While a 3D implementation of VQVAE2 was originally considered, implementing a 3D PixelCNN model seemed impossible, hence the decision was made to extract 2D slices from the MRI scans. A single slice was extracted from the centre of each scan. The MRIs were not all oriented the same, and so slices are a mixture of saggital, axial and coronal.

These slices were split 80/20 into main/test sets, and the main set was further split 80/20 training/validation sets. This split was used to ensure that there was enough data in the test set to ensure diversity of images, but also to ensure enough remained in the training set. Since the dataset size is very small, more test data may negatively impact performance due to lack of diversity in training.

## Setup
### Dependencies
To use the models the following dependencies are required:
```
pytorch >= 1.10
```

To run the evaluation notebook the following additional dependencies are required:
```
torchvision >= 0.9
numpy >= 1.20
matplotlib >= 3.4
pillow >= 8.3
tqdm >= 4.62
python-dotenv >= 0.19
nibabel >= 3.2
```

To run the training scripts the following additional dependencies are required:
```
scipy >= 1.7
```
### Environment File
The scripts use a dotenv file to handle some file paths for data. Create a `.env` file in the root directory and add the following lines:
```
OASIS_PATH = /path/to/oasis/t1w/scans
SLICES_PATH = /path/to/save/slices
```

### Building the Dataset
To build the 2D dataset from the OASIS scans, run `extract_slices.py`.
```
python extract_slices.py
```

## Using the Models
### Evaluating the Pretrained Models
To evaluate the pretrained models, including the SSIM score for VQVAE2 as well as the PixelCNN generator, step through `evaluate.ipynb`.

### Training Scripts
To use the training scripts provided to train the models, first train the VQVAE model. This must be done using PyTorch's Distributed Data Parallel launch script:
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=<num_gpus> train_vqvae.py --batch <batch_size> --dataset oasis --epochs <num_epochs> --savename <vqvae_file_name>
```

Next, train the top-level PixelCNN model using the trained VQVAE:
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=<num_gpus> train_pixel_cnn.py --batch <batch_size> --dataset oasis --epochs <num_epochs> --savename <file_name> -- level top --vqvae <vqvae_file_name>
```

Finally, train the bottom-level PixelCNN model:
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=<num_gpus> train_pixel_cnn.py --batch <batch_size> --dataset oasis --epochs <num_epochs> --savename <file_name> -- level bottom --vqvae <vqvae_file_name>
```

#### A Note on `trainer.py`
This file implements a generic PyTorch model trainer and is subclassed for use in the training scripts to train the individual models used here.

## References
> [1] A. Razavi, A. van den Oord, O. Vinyals, “Generating Diverse High-Fidelity Images with VQ-VAE-2,” in NIPS, Vancouver, Canada, 2019.

> [2] A. van den Oord, O. Vinyals, K. Kavukcuglu, “Neural Discrete Representation Learning,” in NIPS, Long Beach, CA, USA, 2017.

> [3] A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, K. Kavukcuoglu, “Conditional Image Generation with PixelCNN Decoders,” in NIPS, Barcelona, Spain, 2016.