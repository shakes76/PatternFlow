# **Downscale_local_mean algorithm in transform module of PatternFlow**


## **The description of the algorithm:**

The downscale_local_mean algorithm could down-sample N-dimensional image by local averaging.  The downscale_local_mean algorithm calculate the local mean of elements in each block of size factors in the input image.


```
def downscale_local_mean(image, factors, cval=0, clip=True)
```

### **Parameters**
#### image : tensor or ndarray
N-dimensional input image.
#### factors : array
Array containing down-sampling integer factor along each axis.
#### cval : float, optional
Constant padding value if image is not perfectly divisible by the integer factors.
#### clip : bool, optional
Unused, but kept here for API consistency with the other transforms in this module. (The local mean will never fall outside the range of values in the input image, assuming the provided cval also falls within that range.)

### **Returns**
#### image : ndarray
Down-sampled image with same number of dimensions as input image. For integer inputs, the output dtype will be float64. 


***

## **How it works:**
In the downscale_local_mean function, people can input two types of image. The first type of image is ndarray and the second type of image is tensor. This function can convert the ndarray image to tensor image.
```
if tf.is_tensor(image):
    print("tensor")
else:
    print("This is an array and convert to tensor")
    image = tf.convert_to_tensor(image)
```

There are three important functions of **Downscale_local_mean algorithm** algorthm:

* block_reduce function
* view_as_blocks function
* as_strided function

### **block_reduce function**
Down-sample image by applying function to local blocks
```
def block_reduce(image, block_size, func=tf.reduce_sum, cval=0)
```

### **view_as_blocks function**
Block view of the input n-dimensional tensor (using re-striding). 
Blocks are non-overlapping views of the input tensor. 

```
def view_as_blocks(arr_in, block_shape)
```

### **as_strided function**
Create a view into the array with the given shape and strides. 

```
def as_strided(x, shape=None, strides=None, writeable=True)
```


### **The process:**
This algorithm use block_reduce function to down-sample input image by applying function to local blocks. And this algorithm use view_as_blocks function and as_strided function to get the blocks. In each block by using view_as_blocks function, this algorithm calculate the mean of this block and return the result of this block.

#### Example for the process of the algorithm:
```
The original image is:

([[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14]])

Padding in block_reduce function:

  [[ 0  1  2  3  4  0]
   [ 5  6  7  8  9  0]
   [10 11 12 13 14  0]
   [ 0  0  0  0  0  0]]

The blocks with the blcok size is (2,3) by using view_as_blocks function and as_strided function are:

[[[[ 0.  1.  2.]
   [ 5.  6.  7.]]

  [[ 3.  4.  0.]
   [ 8.  9.  0.]]]


 [[[10. 11. 12.]
   [ 0.  0.  0.]]

  [[13. 14.  0.]
   [ 0.  0.  0.]]]]

The result after calculating the mean in block_reduce function:

array([[3.5, 4. ],
       [5.5, 4.5]])

```

***
#### Example for an input image:
Downscale serves the purpose of down-sampling an n-dimensional image by integer factors using the local mean on the elements of each block of the size factors given as a parameter to the function.

The original image is:
![Rocketgray](https://user-images.githubusercontent.com/41613728/66882185-c6f92000-f00c-11e9-8589-c7721f5ed025.png)

The test file is:

```
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage import data, color
from downscale_local_mean import downscale_local_mean


def main():
    """
    get the image from the skimage library
    using the downscale_local_mean to downscaled the image
    show the original and downscaled image
    """
    #get image
    image = color.rgb2gray(data.rocket())
    

    #for the version, we can use tf.compat.v1.Session to ignore the warning 
    tf.InteractiveSession()

    #downscaled the image(if the image is ndarray)
    #image_downscaled = downscale_local_mean(image, (4, 3))

    #downscaled the image(if the image is nd tensor)
    image_tf = tf.convert_to_tensor(image)

    image_downscaled = downscale_local_mean(image_tf, (4, 3))

    fig, axes = plt.subplots(nrows=1, ncols=2)

    #show the original and downscaled images
    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(image_downscaled, cmap='gray')
    ax[1].set_title("Downscaled image (no aliasing)")

    ax[0].set_xlim(0, 630)
    ax[0].set_ylim(400, 0)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

```

The result is:
![Result_1](https://user-images.githubusercontent.com/41613728/66882252-07589e00-f00d-11e9-9f43-27cf14a27a5e.png)




