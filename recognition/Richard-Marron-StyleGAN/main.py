import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import tensorflow as tf
import PIL
import numpy as np
import pathlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(tf.__version__)
    
    img_dir = pathlib.Path("C:/Users/Richard/Desktop/Sem2 2021/COMP3710/Data/AKOA_Analysis")
    
    images = tf.keras.utils.image_dataset_from_directory(img_dir,
                                                         labels=None,
                                                         color_mode="grayscale",
                                                         image_size=(260, 228),
                                                         seed=123,
                                                         subset="training",
                                                         validation_split=0.2)
    
    images = images.map(lambda x: (tf.divide(x, 255)))
    
    for e in images:
        plt.imshow(e[0].numpy(), cmap="gray")
        break
    #plt.imshow(im[0].numpy().astype("uint8"))
    plt.show()
    