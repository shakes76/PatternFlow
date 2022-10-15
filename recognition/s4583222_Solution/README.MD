# **Stable Diffusion on the OASIS Brain Dataset**
**COMP3710 Report by Thomas Jellett (45832220)**

## **Stable Diffusion:**
Diffusion Models are a generative model. They can be used to generate new data, similar to the data it was trained on. Diffusion is the process of detroying the training data through successive iterations of Gaussian noise. The Diffusion model then learns how to recover the data by reversing the noise. We can then generate images by passing randomly sampled noise (O'Connor, 2022). 

There are two main processes in diffusion, the ***Forward Diffusion Process*** and the ***Reverse Diffusion Process***. The math for both processes is very complex and long so I recommend either checking out the *Denoising Diffusion Probabilistic Models* [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/pdf/2006.11239.pdf) paper or this [*Repository*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) as both give an indepth explaination on the derivation of the maths. But to summarize, 
- **Forward Diffusion Process** is when noise is gradually added to the image until you get pure noise. And,
- **Reverse Diffusion Process** is the learned process where a neural network is trained to denoise the image from pure noise to an actual image.

![Generated_brain](../Git%20images/455_0_0_backward.png)


## **File Structure:**
The code was written in python and has been split up into 4 files. Each file serves a different purpose.

- **modules.py**: Contains the code for the UNET Model and Diffusion process.
- **dataset.py**: Contains the functions and classes used to load and preprocess the image data.
- **train.py**:  Contains the code for training, testing and saving the model.
- **predict.py**: The predict file loads up a previously trained model and generates results.

## **Loading Training Data:**
To load the Oasis Brain 
## **UNET Model:**

## **Diffusion:**

## **Training:**

## **Issues:**

## **Predict/Results:**

## **Dependencies:**
The following packages were installed on conda and were used to run the code.
- _libgcc_mutex             0.1                        main
- _openmp_mutex             5.1                       1_gnu
- blas                      1.0                         mkl
- brotli                    1.0.9                h5eee18b_7
- brotli-bin                1.0.9                h5eee18b_7
- brotlipy                  0.7.0           py39h27cfd23_1003
- bzip2                     1.0.8                h7f98852_4
- ca-certificates           2022.07.19           h06a4308_0
- certifi                   2022.9.24        py39h06a4308_0
- cffi                      1.15.1           py39h74dc2b5_0
- charset-normalizer        2.0.4              pyhd3eb1b0_0
- colorama                  0.4.4              pyhd3eb1b0_0
- conda                     22.9.0           py39h06a4308_0
- conda-content-trust       0.1.1              pyhd3eb1b0_0
- conda-package-handling    1.9.0            py39h5eee18b_0
- cryptography              37.0.1           py39h9ce1e76_0
- cudatoolkit               11.3.1               h2bc3f7f_2
- cycler                    0.11.0             pyhd3eb1b0_0
- dbus                      1.13.18              hb2f20db_0
- expat                     2.4.9                h6a678d5_0
- ffmpeg                    4.3.2                hca11adc_0
- fontconfig                2.13.1               h6c09931_0
- fonttools                 4.25.0             pyhd3eb1b0_0
- freetype                  2.11.0               h70c0345_0
- future                    0.18.2           py39h06a4308_1
- giflib                    5.2.1                h7b6447c_0
- glib                      2.69.1               h4ff587b_1
- gmp                       6.2.1                h58526e2_0    
- gnutls                    3.6.13               h85f3911_1    
- gst-plugins-base          1.14.0               h8213a91_2
- gstreamer                 1.14.0               h28cd5cc_2
- icu                       58.2                 he6710b0_3
- idna                      3.3                pyhd3eb1b0_0
- intel-openmp              2021.4.0          h06a4308_3561
- jpeg                      9e                   h7f8727e_0
- kiwisolver                1.4.2            py39h295c915_0
- krb5                      1.19.2               hac12032_0
- lame                      3.100             h7f98852_1001    
- lcms2                     2.12                 h3be6417_0
- ld_impl_linux-64          2.38                 h1181459_1
- lerc                      3.0                  h295c915_0
- libblas                   3.9.0            12_linux64_mkl    
- libbrotlicommon           1.0.9                h5eee18b_7
- libbrotlidec              1.0.9                h5eee18b_7
- libbrotlienc              1.0.9                h5eee18b_7
- libcblas                  3.9.0            12_linux64_mkl    
- libclang                  10.0.1          default_hb85057a_2
- libdeflate                1.8                  h7f8727e_5
- libedit                   3.1.20210910         h7f8727e_0
- libevent                  2.1.12               h8f2d780_0
- libffi                    3.3                  he6710b0_2
- libgcc-ng                 11.2.0               h1234567_1
- libgomp                   11.2.0               h1234567_1
- liblapack                 3.9.0            12_linux64_mkl    
- libllvm10                 10.0.1               hbcb73fb_5
- libpng                    1.6.37               hbc83047_0
- libpq                     12.9                 h16c4e8d_3
- libprotobuf               3.16.0               h780b84a_0    
- libstdcxx-ng              11.2.0               h1234567_1
- libtiff                   4.4.0                hecacb30_0
- libuuid                   1.0.3                h7f8727e_2
- libwebp                   1.2.4                h11a3e52_0
- libwebp-base              1.2.4                h5eee18b_0
- libxcb                    1.15                 h7f8727e_0
- libxkbcommon              1.0.1                hfa300c1_0
- libxml2                   2.9.14               h74e7548_0
- libxslt                   1.1.35               h4e12654_0
- lz4-c                     1.9.3                h295c915_1
- matplotlib                3.5.2            py39h06a4308_0
- matplotlib-base           3.5.2            py39hf590b9c_0
- mkl                       2021.4.0           h06a4308_640
- mkl-service               2.4.0            py39h7f8727e_0
- mkl_fft                   1.3.1            py39hd3c417c_0
- mkl_random                1.2.2            py39h51133e4_0
- munkres                   1.1.4                      py_0
- ncurses                   6.3                  h5eee18b_3
- nettle                    3.6                  he412f7d_0    
- ninja                     1.10.2               h06a4308_5
- ninja-base                1.10.2               hd09550d_5
- nspr                      4.33                 h295c915_0
- nss                       3.74                 h0370c37_0
- numpy                     1.23.1           py39h6c91a56_0
- numpy-base                1.23.1           py39ha15fc14_0
- openh264                  2.1.1                h4ff587b_0
- openssl                   1.1.1q               h7f8727e_0
- packaging                 21.3               pyhd3eb1b0_0
- pcre                      8.45                 h295c915_0
- pillow                    9.0.1            py39h22f2fdc_0    
- pip                       22.2.2           py39h06a4308_0
- ply                       3.11             py39h06a4308_0
- pycosat                   0.6.3            py39h27cfd23_0
- pycparser                 2.21               pyhd3eb1b0_0
- pyopenssl                 22.0.0             pyhd3eb1b0_0
- pyparsing                 3.0.9            py39h06a4308_0
- pyqt                      5.15.7           py39h6a678d5_1
- pyqt5-sip                 12.11.0          py39h6a678d5_1
- pysocks                   1.7.1            py39h06a4308_0
- python                    3.9.12               h12debd9_0
- python-dateutil           2.8.2              pyhd3eb1b0_0
- python_abi                3.9                      2_cp39    
- pytorch                   1.12.1          py3.9_cuda11.3_cudnn8.3.2_0
- pytorch-mutex             1.0                        cuda    
- qt-main                   5.15.2               h327a75a_7
- qt-webengine              5.15.9               hd2b0992_4
- qtwebkit                  5.212                h4eab89a_4
- readline                  8.1.2                h7f8727e_1
- requests                  2.28.1           py39h06a4308_0
- ruamel_yaml               0.15.100         py39h27cfd23_0
- setuptools                63.4.1           py39h06a4308_0
- sip                       6.6.2            py39h6a678d5_0
- six                       1.16.0             pyhd3eb1b0_1
- sleef                     3.5.1                h7f98852_1    
- sqlite                    3.39.3               h5082296_0
- tk                        8.6.12               h1ccaba5_0
- toml                      0.10.2             pyhd3eb1b0_0
- toolz                     0.11.2             pyhd3eb1b0_0
- torchaudio                0.12.1               py39_cu113    
- torchvision               0.13.1               py39_cu113    
- tornado                   6.2              py39h5eee18b_0
- tqdm                      4.64.1           py39h06a4308_0
- typing-extensions         4.3.0            py39h06a4308_0
- typing_extensions         4.3.0            py39h06a4308_0
- tzdata                    2022c                h04d1e81_0
- urllib3                   1.26.11          py39h06a4308_0
- wheel                     0.37.1             pyhd3eb1b0_0
- x264                      1!161.3030           h7f98852_1    
- xz                        5.2.6                h5eee18b_0
- yaml                      0.2.5                h7b6447c_0
- zlib                      1.2.12               h5eee18b_3
- zstd                      1.5.2                ha4553b6_0

## **References:**

(Nir Barazida, 2022) Towards Data Science (2022) Stable Diffusion: Best Open Source Version of DALL·E 2. Available at: https://towardsdatascience.com/stable-diffusion-best-open-source-version-of-dall-e-2-ebcdf1cb64bc

(O'Connor, 2022) Assembly AI (2022). Introduction to Diffusion Models for Machine Learning. Available at: https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/.

‌