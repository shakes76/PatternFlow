# **Variant of Stable Diffusion on the OASIS Brain Dataset**
**COMP3710 Report by Thomas Jellett (45832220)**

## **Stable Diffusion:**
Diffusion Models are a generative model. They can be used to generate new data, similar to the data it was trained on. Diffusion is the process of detroying the training data through successive iterations of Gaussian noise. The Diffusion model then learns how to recover the data by reversing the noise. We can then generate images by passing randomly sampled noise (O'Connor, 2022). 

There are two main processes in diffusion, the ***Forward Diffusion Process*** and the ***Reverse Diffusion Process***. The math for both processes is very complex and long so I recommend either checking out the *Denoising Diffusion Probabilistic Models* [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/pdf/2006.11239.pdf) paper or this [*Repository*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) as both give an indepth explaination on the derivation of the maths. But to summarize, 
- **Forward Diffusion Process** is when noise is gradually added to the image until you get pure noise. And,
<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/0_0_0_backward_smaller.png?raw=true>
</p>

- **Reverse Diffusion Process** is the learned process where a neural network is trained to denoise the image from pure noise to an actual image.

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/455_0_0_backward_smaller.png?raw=true>
</p>

Stable diffusion models are diffusion models but have a cross-attention layer in the model architecture. By introducting this layer, the generator can be turned into a more flexible generator (Rombach, 2022). However, the model created in this report differs from stable diffusion in that we are only reproducing images of brains so we do not need to be flexible in the images that we are generating. Thus, we can get rid of some complexity in our architecture.

## **File Structure:**
The code was written in python and has been split up into 4 files. Each file serves a different purpose.

- **modules.py**: Contains the code for the UNET Model.
- **dataset.py**: Contains the functions and classes used to load and preprocess the image data.
- **train.py**:  Contains the code for training (forward process) and saving the model.
- **predict.py**: The predict file loads up a previously trained model and generates results (reverse process).

## **Dataset:**
When the dataset is installed, the folder structure is 

- **keras_png_slices_seg_test** 
- **keras_png_slices_seg_train**
- **keras_png_slices_seg_validate**
- **keras_png_slices_test**
- **keras_png_slices_train**
- **keras_png_slices_validate**

The seg images were discarded since only images of brain scans are required to be generated. Therefore, the contents of **keras_png_slices_test**, **keras_png_slices_train**, and **keras_png_slices_validate** were copied into a folder called **Images**. This **Images** consisted of a **Train** subfolder which stored the 11,328 brain images.

The Oasis Brain dataset was loaded using the ```get_data_loader(training_images, batch, workers, pin_mem)``` function.
```python
def get_data_loader(training_images, batch, workers, pin_mem):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    training_dataset = BrainDataset(image_directory=training_images, transform=transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=workers, pin_memory=pin_mem, drop_last=True)
    return training_loader    
```
The image directory is given to the function (as well as batch size, number of workers, and pin memory). The BrainDataset class then preprocesses the images. The images are first opened as RGB using the Pillow package.

```python
from PIL import Image
image = Image.open(image_path).convert("RGB")
```
The images are then transformed.
```python
transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])
```
The images original size is 256x256, but because of memory issues the images were resized to 64x64 and the later to 128x128. The images are randomly flipped horizontally. The Pil images are convert to tensors. Finally they are also normalized.

from ```torch.utils.data```, the transformed images are then passed to ```DataLoader```.
```python
training_dataset = BrainDataset(image_directory=training_images, transform=transforms)
training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=workers, pin_memory=pin_mem, drop_last=True) 
```
The batch_size used was 64. The training can be sped up by increasing the batch size, however, if (number of images / batch_size) does not result in a real integer then the model will fail when training. To fix this set ```drop_last=True``` in the ```DataLoader```.

## **Modules:**
Diffusion models are highly flexible in the sense that they allow any architecture as long as the dimensions of the input and the output remain the same. Therefore, the model chosen is a modified UNET architecture to predict the amount of noise in a given image. The model follows the basic architecture of a simple UNET, but within each level of the UNET we specify to the model the sinusoidal position embedding. 

The UNET chanels used was ```[64, 128, 256, 512, 1024]```. Increasing this would result in more effective training, but doing so might result in memory issues whilst training. Such issues occured whilst training. Thus, the size of the model had to be decreased to the chanels above. 

The total number of trainable parameters is
```
Num params:  31141859
```

```python
t = self.time_embedding(t)

x1 = self.first_level(x, t)
x1_skip = x1

x2 = self.pool(x1)
x2 = self.second_level(x2, t)
x2_skip = x2

x3 = self.pool(x2)
x3 = self.third_level(x3, t)
x3_skip = x3

x4 = self.pool(x3)
x4 = self.fourth_level(x4, t)
x4_skip = x4

x_bottle = self.pool(x4)
x_bottle = self.bottle_neck(x_bottle, t)

x5 = self.fifth_level_up(x_bottle)        
x5 = torch.cat([x4_skip, x5], dim = 1)
x5 = self.fifth_level(x5, t)

x6 = self.sixth_level_up(x5) 
x6 = torch.cat([x3_skip, x6], dim = 1)
x6 = self.sixth_level(x6, t)

x7 = self.seventh_level_up(x6) 
x7 = torch.cat([x2_skip, x7], dim = 1)
x7 = self.seventh_level(x7, t)

x8 = self.eigth_level_up(x7) 
x8 = torch.cat([x1_skip, x8], dim = 1)
x8 = self.eigth_level(x8, t) 

output_x = self.last_conv(x8)
```
Each level of the UNET consists of a "Block", and within each "Block" there is a convolution increase the number of channels (i.e. 126x126x3 -> 126x126x64), followed by a batch normalisation and a ReLU activation layer. Then we add the sinusoidal position embedding. Finally a second convolution layer, followed by another batch normalisation and ReLU activation layer. 

The model contraction path of the UNET is four levels of a block, skip connection and a max pool. The expantion path is then an upward convultion (upsample), skip connection concatenation and a block. 


## **Training:**

The training file contains the functions for the noising (forward process), training, ploting the loss, and saving. For training, the maximum time step was set to 600 (```NOISE_STEP = 600```), this is much smaller than the suggested 1000 in the original paper. 600 was chosen because, as seen in the image below, after about 600~700 timesteps the original images is already complete destroyed. Thus, by making the timestep 1000 it would be harder to training the model to denoise the image. 

<p align = "center">
<img width = "745" height = "215" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/test_2.png?raw=true>
</p>

For training the loss function used was ```l1_loss``` and the optimizer used was ```optim.Adam```. 

```python
vloss = []

noisy_image, actual_noise = forward_diffusion_to_image(image, t)
predicted_noise = model(noisy_image, t)
loss = F.l1_loss(actual_noise, predicted_noise)
vloss.append(loss.item()) 
```
### **Plotting:**
The loss is appended to a list ```vloss```, this list is then plotted every 10 iterations within every epoch. The model was trained for approximately 205 epochs. The resulting plot of the loss is as below.

<p align = "center">
<img width = "640" height = "480" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/loss_plot.png?raw=true>
</p>
(note: the x-axis is all the iterations performed. Epoch = iteration/number of iterations in a batch, i.e. 205 = 36285/177)

It can be observed from the plot that the loss rapidly optimizes itself within 5-10 epochs, after that it just fine tunes itself.
This can also be seen from the ```.out``` log below.

```
Epoch: 0 | iteration: 1 | Loss: 0.8565319776535034
Epoch: 5 | iteration: 1 | Loss: 0.11129455268383026
Epoch: 10 | iteration: 1 | Loss: 0.10953794419765472
Epoch: 15 | iteration: 1 | Loss: 0.09719376266002655
Epoch: 20 | iteration: 1 | Loss: 0.07053589820861816
Epoch: 25 | iteration: 1 | Loss: 0.07387645542621613
Epoch: 30 | iteration: 1 | Loss: 0.07491575181484222
Epoch: 35 | iteration: 1 | Loss: 0.060492269694805145
Epoch: 40 | iteration: 1 | Loss: 0.06895732879638672
Epoch: 45 | iteration: 1 | Loss: 0.06820546090602875
Epoch: 50 | iteration: 1 | Loss: 0.06779824942350388
Epoch: 55 | iteration: 1 | Loss: 0.054499588906764984
Epoch: 60 | iteration: 1 | Loss: 0.05144139379262924
Epoch: 65 | iteration: 1 | Loss: 0.06103380024433136
Epoch: 70 | iteration: 1 | Loss: 0.045349933207035065
...
Epoch: 200 | iteration: 1 | Loss: 0.05957743152976036
Epoch: 205 | iteration: 1 | Loss: 0.03544085845351219
```

### **Saving:**
Within the training loop there are two instances where the model saves. The first instance occurs every 10 iterations in each epoch, the second occurs every 5 epochs in instance 1. This will result in slower training but they are saved to different files at different instances because if one model fails to save correctly there is at another model you can load up from 5 epochs ago. 
```python
if epoch % 1 == 0 and i % 10 == 0:
    torch.save(model.state_dict(), "Model_2.pt")
    plot_v = torch.tensor(vloss, device='cpu')
    plt.title('Stable Diffusion Loss')
    plt.plot(plot_v, color = 'blue')
    plt.savefig('loss_plot')

if epoch % 5 == 0 and i == 1:
    print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
    torch.save(model.state_dict(), "Model_backup.pt")
```
### **Loading Model:**
If the training was to abruptly stop, simply set 
```python
LOAD_MODEL = True
```
and then set the following to read from the model file you wish to continue from
```python
if LOAD_MODEL:
    model.load_state_dict(torch.load("Model_2.pt"))
```
## **Predict:**
The predict file will generate images based on the weights of the trained model. To get started simply load up the model to load and change the number of image to be generated.
```python
model = UNETModel().to(DEVICE)
model.load_state_dict(torch.load("Model_2.pt"))
gen_images = 1
```
The denoising process is then applied to a random distribution of noise and an image is saved. When finished, ```Done!``` will be printed to ```.out```. 

Here are some of the results of the trained model.

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/1_0_0_backward.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/nice.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/2_0_0_backward_colour.png?raw=true>
</p>

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/3_0_0_backward_colour.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/12_0_0_backward_colour.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/21_0_0_backward_colour.png?raw=true>
</p>

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/28_0_0_backward_colour.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/15_0_0_backward_colour.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/41_0_0_backward_colour.png?raw=true>
</p>

The model is getting the shape of the brain to be correct in most cases, but the colour is sometimes off (e.g. a pink brain). A "bandaid" solution is to open the image after saving the colour version and then using the Pillow package open the image in grayscale ```.convert("L")``` and save the image again. 
```python
im = Image.open(f"{epoch}_{i}_{disc}_colour.png").convert("L")
plt.imshow(im, cmap='gray')
plt.savefig(f"{epoch}_{i}_{disc}_gray")
```
The result in doing so is as below.

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/3_0_0_backward_gray.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/12_0_0_backward_gray.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/21_0_0_backward_gray.png?raw=true>
</p>

<p align = "center">
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/28_0_0_backward_gray.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/15_0_0_backward_gray.png?raw=true>
&nbsp;
<img width = "256" height = "256" src =https://github.com/s4583222/PatternFlow/blob/master/recognition/s4583222_Solution/Git%20images/41_0_0_backward_gray.png?raw=true>
</p>

As observed from the images above, the images generated by the model are clearly brains, and although the results aren't identical to the Oasis data it can be concluded that they are pretty close/reasonably clear.  

## **Suggestions for Improvement:**
To address the "bandaid" solution it is suggested that inorder to not produce coloured brains, in the ```BrainDataset``` class in **dataset.py**, instead of loading the data in as RGB, instead load it in as L (grayscale).
```python
image = Image.open(image_path).convert("RGB")
```
becomes
```python
image = Image.open(image_path).convert("L")
```
A few other lines will need to be changed within the model such as the **first** and **final** convolution. But doing so will speedup training time and possibly produce more optimal results.

Better results could also possibly be produced with larger images. Instead of 128x128 try 256x256 (original image size). In order to not have memory issues, the batch size may need to be decreased and doing so would result in a slower training speed, but should produce results more identical to the training set.

## **Replication of Results:**
The model was trained using UQ's Rangpur. If able to sign ssh into Rangpur, the results can be replicated by placing the **modules.py**, **dataset.py**, **train.py**, **predict.py** files into a directory with the **Images** folder described earlier. 

Two ```.sh``` files are going to be created to run the jobs on the cluster. One for **train.py**, and one for **predict.py**. The contents of the two files are as below.
### **Train.sh:**
```
#!/bin/bash
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1
#SBATCH --job-name="StableDiff"
#SBATCH --output=R-%x.out
#SBATCH --error=R-%x.err

module purge
module load cuda/11.4
source YOUR_DIRECTORY/miniconda3/bin/activate 
python3 train.py
```

### **Predict.sh:**
```
#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --job-name="Predict"
#SBATCH --output=Pred-%x.out
#SBATCH --error=Pred-%x.err

module purge
module load cuda/11.4
source YOUR_DIRECTORY/miniconda3/bin/activate 
python3 predict.py
```

In the terminal enter
```
sbatch Train.sh
```
and once enough epochs have passed, in the terminal enter
```
sbatch Predict.sh
```

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

(Rombach, 2022) Rombach, R., Blattmann, A., Lorenz, D., Esser, P. and Ommer, B. (n.d.). High-Resolution Image Synthesis with Latent Diffusion Models. Available at: https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf.
‌
## **Acknowledgments:**

DeepFindr, Diffusion Models from scratch in PyTorch
(DeepFindr, 2022) Diffusion Models from scratch in PyTorch.
Available at: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=HhIgGq3za0yh

(Rogge, 2022) Niels Rogge and Kashif Rasul. Hugging Face (2022) The Annotated Diffusion Model
Available at: https://huggingface.co/blog/annotated-diffusion 