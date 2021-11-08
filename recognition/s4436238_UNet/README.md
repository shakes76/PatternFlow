# README - Improved UNet on ISIC Dataset
### Molly Parker s4436238

The UNet is a convolutional neural network that gets its name from its shape. The U-shaped network aims to effectively and efficiently segment images and output a segmentation map (i.e. a black and white image showing only the important boundaries from the original images). The downsampling portion of the network works to identify the key segments of the image, and the upsampling portion increases the resolution of the resulting map. 

![UNet architecture by Jeremy Zhang](https://miro.medium.com/max/1838/1*f7YOaE4TWubwaFF7Z1fzNw.png)

This model aims to implement the Improved UNet over the ISIC 2018 dataset in order to identify the boundaries of images of skin lesions. 


ISIC: https://challenge2018.isic-archive.com/

UNet: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/#:~:text=The%20u%2Dnet%20is%20convolutional,and%20precise%20segmentation%20of%20images.&text=U%2Dnet%20architecture%20(example%20for,on%20top%20of%20the%20box.

Improved UNet: https://arxiv.org/pdf/1802.10508v1.pdf
