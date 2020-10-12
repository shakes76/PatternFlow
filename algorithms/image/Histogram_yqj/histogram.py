from skimage import data, exposure, img_as_float
from skimage import img_as_ubyte
import tensorflow as tf
import matplotlib.pyplot as plt


def histogram(image, nbins=256, source_range='image', normalize=False):
  '''
  Visualize the image to histogram of the numbers of pixels in order to each bin of pixel values

  Args:
    image(8-bit uint): Input image
    nbins(Int): Number of bins in histogram
    source_range(str): 'image’ (default) determines the range from the input image
    normalize(bool): If True, normalize the histogram by the sum of its values.
  '''
  with tf.Session() as sess:
    image = tf.convert_to_tensor(image, dtype = tf.float32) # Change image to tensorflow
    sh = sess.run(tf.shape(image))
    if len(sh) == 3 and sh[-1] < 4:
      warn("This might be a color image. The histogram will be "
            "computed on the flattened image. You can instead "
            "apply this function to each color channel.")
    image = tf.reshape(image,[-1,1]) # 512**2 ==> (262144,)
    value_range = [0.0, 256.0] 
    new_values = image
    hist = sess.run(tf.histogram_fixed_width ( new_values.eval() ,value_range ,nbins = 256 ) ) #Show number of values in each bin. 256 (0~256).
    hist_centers = [i for i in range(256)] # list,(0~255)
  sess.close()
  return hist, hist_centers



