# Stable Diffusion on the ADNI dataset

## Design Layout

`imports.py` contains the basic imports requried throughout our implementation.

`dataset.py` contains the data loader for loading and preprocessing the ADNI dataset.

`modules.py` contains the implementation of the stable diffusion model.

`train.py` contains the code for training, validating, testing and saving of the model.

`predict.py` contains example usage of the trained stable diffusion model.

## Data Loading and Preprocessing

To handle data loading, we use `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. 
`Dataset` allows for the storage of our image data, while `Dataloader` wraps an iterable around the pre-defined `Dataset`, allowing simplified access to our image data such as when training our model.
In terms of preprocessing, we apply the transformation `torchvision.transforms.ToTensor()` within our `Dataset`, which converts a `PIL` Image with shape (HxWxC) in the range [0,255] to a  `torch.FloatTensor` with shape (CxHxW) and range [0,1]. This portion of our data loading implementation can be found within `dataset.py`.

Before passing our (batches of) image data into our model, one final set of transformations is applied. These are taken from `torchvision.transforms`, and include: `Grayscale` (ensures the images only have one colour channel), `Resize` (ensure all images have same size), `CenterCrop`, and a linear scaling into [-1,1] via an applying `Lambda(lambda t: (t * 2) - 1)`.

Since our aim is to generate a number of images via our Stable Diffusion model, and given that our dataset is split into brain MRI images with and without Alzheimer's disease, we chose to simply combine both partitions to increase the number of training images available. Since we do not implement a "Test" module, as described in "Validation" below, our validation dataset includes a small percentage of the "test" images within the ADNI dataset.

## Model Details

Our model of choice is an implementation of the Stable Diffusion, as proposed in the paper https://arxiv.org/abs/2112.10752, and as shown visually below.

![Stable Diffusion diagram](images/stable_diffusion_diagram.png)

Stable Diffusion is based on Latent Diffusion, in which the network is trained to "denoise" random Gaussian noise. In Latent Diffusion, instead of working in pixel space, our (UNet) model is trained to generate latent representations of given images -- and after training, we can use this same model, and a "denoising" process, to recieve images visually similar to those in which the model was trained with.

It is important to note that our implementation does not include "conditioning", as well as "crossattention", as depicted in the above diagram. Since our aim is to reproduce images of a singly type (i.e. in our case, images of a segmented brain), we concluded that the addition of "conditioning" and "crossattention" was not required.

Hence, our implementation is more aligned with the "original" latent diffusion paper https://arxiv.org/abs/2006.11239 (which likewise does not implement "conditioning" or "crossattention"), as reference in the Stable Diffusion paper.

## Training

Throughout training, we implemented F1 loss via `torch.nn.functional.l1_loss`.
Shown below is the generated loss curve, training for 50 epochs total.

![Training Loss graph](images/loss_curve.png)




## Validation

Validation was performed throughout the training process in `train.py`. In particular, since our model/network is not a 'clasiffier' network, we cannot simply count the numbre of correct 'guesses' that our model predicts. Instead, we can simply take the `get_loss` function used to train the model, and on another (seperate) dataset (i.e. one that the model has not been trained on), apply the `get_loss` function here and return the validation loss, which can then be plotted alongside the training loss as shown below.

![Training/Validation Loss graph](images/loss_val_curve.png)

Note that it doesn't follow with our model type to have 'Test' functionality, on top of our 'Train' and 'Validate' functionality, again since our model is not a classification network.


## Usage

We can use `predict.py` to visualise the backwards diffusion process via
```python
model = torch.load("DiffusionModel")
reverse_diffusion(model, plot="plot_diffusion_process")
```
as shown below.

![Reverse diffusion process plot](images/diffusion_process.png)

We can use `predict.py` to visualise a number of final denoised images, via the backwards diffusion process, via
```python
model = torch.load("DiffusionModel")
reverse_diffusion(model, plot="image_grid")
```
as shown below.

![Reverse diffusion image grid plot](images/image_grid.png)

We can also use `predict.py` to visualise the backwards diffusion process, on a single sample, via the generation of a `.gif` through
```python
model = torch.load("DiffusionModel")
reverse_diffusion(model, plot="image_gif")
```
as shown below.

![Reverse diffusion gif](images/image_gif.gif)

## Dependencies

The following table was generate via the `conda list` within our conda environment. Note that the majority of these were installed via `conda install pytorch torchvision torchaudio cudatoolkit=11.7.0 -c pytorch -c conda-forge`, as suggest via the home page of `https://pytorch.org/`.

For quick reference, note the main packages we have include `python=3.10.6`, `pytorch==1.11.0`, `matplotlib=3.6.0` and `pillow=9.2.0`.

| Name        |            Version     |
|-------------|------------------------|       
| _libgcc_mutex |            0.1        |    
 | _openmp_mutex     |        4.5           |  
 | alsa-lib     |           1.2.7.2          |
 | aom     |                3.5.0 |
 | attr     |               2.5.1  |         
 | brotli     |             1.0.9     |        
 | brotli-bin     |         1.0.9   |        
 | brotlipy     |           0.7.0    |      
 | bzip2     |              1.0.8     |       
 | ca-certificates |          2022.9.24 |         
 | certifi     |            2022.9.24 |
 | cffi     |               1.15.1    |
 | charset-normalizer  |      2.1.1     |
 | colorama     |           0.4.5     |
 | conda     |              22.9.0    |
 | conda-package-handling |   1.9.0     |
 | contourpy     |          1.0.5     |
 | cryptography     |       37.0.4    |
 | cudatoolkit     |        11.7.0    |
 | cudnn     |              8.4.1.50  |
 | cycler     |             0.11.0    |
 | dbus     |               1.13.6    |
 | expat     |              2.4.9     |
 | ffmpeg     |             4.4.2      |
 | fftw     |               3.3.10      | 
 | font-ttf-dejavu-sans-mono|  2.37      |
 | font-ttf-inconsolata    |  3.000     |
 | font-ttf-source-code-pro|  2.038     |
 | font-ttf-ubuntu    |       0.83      |
 | fontconfig     |         2.14.0    |
 | fonts-conda-ecosystem |    1        | 
 | fonts-conda-forge   |      1         |
 | fonttools     |          4.37.4    |
 | freetype     |           2.12.1    |
 | gettext     |            0.19.8.1  |
 | glib     |               2.74.0    |
 | glib-tools     |         2.74.0    |
 | gmp     |                6.2.1     |
 | gnutls     |             3.7.8     |
 | gst-plugins-base    |      1.20.3   | 
 | gstreamer     |          1.20.3    |
 | icu     |                70.1      |
 | idna     |               3.4       |
 | jack     |               1.9.18    |
 | jpeg     |               9e        |
 | keyutils     |           1.6.1     |
 | kiwisolver     |         1.4.4     |
 | krb5     |               1.19.3    |
 | lame     |               3.100     |
 | lcms2     |              2.12      |
 | ld_impl_linux-64  |        2.36.1   | 
 | lerc     |               4.0.0     |
 | libblas     |            3.9.0     |
 | libbrotlicommon  |         1.0.9     |
 | libbrotlidec     |       1.0.9     |
 | libbrotlienc     |       1.0.9     |
 | libcap     |             2.65      |
 | libcblas     |           3.9.0     |
 | libclang     |           14.0.6     |  
 | libclang13     |         14.0.6      | 
 | libcups     |            2.3.3       |
 | libdb     |              6.2.32      |
 | libdeflate     |         1.14        |
 | libdrm     |             2.4.113     |
 | libedit     |            3.1.20191231|
 | libevent     |           2.1.10      |
 | libffi     |             3.4.2       |
 | libflac     |            1.3.4       |
 | libgcc-ng     |          12.1.0      |
 | libgfortran-ng   |         12.1.0      |
 | libgfortran5     |       12.1.0      |
 | libglib     |            2.74.0      |
 | libiconv     |           1.17        |
 | libidn2     |            2.3.3       |
 | liblapack     |          3.9.0       |
 | libllvm14     |          14.0.6      |
 | libnsl     |             2.0.0       |
 | libogg     |             1.3.4       |
 | libopus     |            1.3.1       |
 | libpciaccess     |       0.16        |
 | libpng     |             1.6.38      |
 | libpq     |              14.5        |
 | libprotobuf     |        3.20.1      |
 | libsndfile     |         1.0.31      |
 | libsqlite     |          3.39.3      |
 | libstdcxx-ng     |       12.1.0      |
 | libtasn1     |           4.19.0      |
 | libtiff     |            4.4.0       |
 | libtool     |            2.4.6       |
 | libudev1     |           249         |
 | libunistring     |       0.9.10      |
 | libuuid     |            2.32.1      |
 | libva     |              2.16.0      |
 | libvorbis     |          1.3.7       |
 | libvpx     |             1.11.0      |
 | libwebp-base     |       1.2.4       |
 | libxcb     |             1.13        |
 | libxkbcommon     |       1.0.3       |
 | libxml2     |            2.9.14      |
 | libzlib     |            1.2.12      |
 | llvm-openmp     |        14.0.4      |
 | magma     |              2.5.4       |
 | matplotlib     |         3.6.0       |
 | matplotlib-base   |        3.6.0       |
 | mkl     |                2022.1.0    |
 | munkres     |            1.1.4       |
 | mysql-common     |       8.0.30      |
 | mysql-libs     |         8.0.30      |
 | nccl     |               2.14.3.1    |
 | ncurses     |            6.3         |
 | nettle     |             3.8.1       |
 | ninja     |              1.11.0      |
 | nspr     |               4.32        |
 | nss     |                3.78        |
 | numpy     |              1.23.3      |
 | openh264     |           2.3.1       |
 | openjpeg     |           2.5.0       |
 | openssl     |            1.1.1q      |
 | p11-kit     |            0.24.1      |
 | packaging     |          21.3        |
 | pcre2     |              10.37       |
 | pillow     |             9.2.0       |
 | pip     |                22.2.2      |
 | ply     |                3.11        |
 | portaudio     |          19.6.0      |
 | pthread-stubs     |        0.4         |
 | pulseaudio     |         14.0        |
 | pycosat     |            0.6.3        |  
 | pycparser     |          2.21      |
 | pyopenssl     |          22.0.0    |
 | pyparsing     |          3.0.9     |
 | pyqt     |               5.15.7    |
 | pyqt5-sip     |          12.11.0   |
 | pysocks     |            1.7.1     |
 | python     |             3.10.6     |  
 | python-dateutil    |       2.8.2     |
 | python_abi     |         3.10      |
 | pytorch     |            1.11.0     |   
 | pytorch-gpu     |        1.11.0      |  
 | qt-main     |            5.15.6    |
 | readline     |           8.1.2     |
 | requests     |           2.28.1    |
 | ruamel_yaml     |        0.15.80    |  
 | setuptools     |         65.4.0    |
 | sip     |                6.6.2     |
 | six     |                1.16.0    |
 | sleef     |              3.5.1     |
 | sqlite     |             3.39.3    |
 | svt-av1     |            1.2.1     |
 | tbb     |                2021.6.0  |
 | tk     |                 8.6.12    |
 | toml     |               0.10.2    |
 | toolz     |              0.12.0    |
 | torchvision     |        0.12.0     |    
 | tornado     |            6.2       |
 | tqdm     |               4.64.1    |
 | typing_extensions |        4.3.0    | 
 | tzdata     |             2022d     |
 | unicodedata2     |       14.0.0    |
 | urllib3     |            1.26.11   |
 | wheel     |              0.37.1    |
 | x264     |               1!164.3095|
 | x265     |               3.5       |
 | xcb-util     |           0.4.0     |
 | xcb-util-image      |      0.4.0    | 
 | xcb-util-keysyms   |       0.4.0     |
 | xcb-util-renderutil  |     0.3.9     |
 | xcb-util-wm     |        0.4.1     |
 | xorg-fixesproto |          5.0      | 
 | xorg-kbproto     |       1.0.7     |
 | xorg-libx11     |        1.7.2     |
 | xorg-libxau     |        1.0.9     |
 | xorg-libxdmcp   |          1.1.3     |
 | xorg-libxext     |       1.3.4     |
 | xorg-libxfixes   |         5.0.3    | 
 | xorg-xextproto   |         7.3.0     |
 | xorg-xproto     |        7.0.31    |
 | xz     |                 5.2.6     |
 | yaml     |               0.2.5     |
 | zstd     |               1.5.2     |