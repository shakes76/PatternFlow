## load modules
import tensorflow as tf
import math
# In[2]:
sess = tf.InteractiveSession()
def iradon(radon_image, theta=None, output_size=None, filter="ramp", interpolation="linear", circle=True):
    radon_image = tf.constant(radon_image, tf.float64)
    
    if tf.rank(radon_image).eval() != 2:
        print(tf.rank(radon_image))
        raise ValueError('The input image must be 2-D')
   
    if theta is None:
        theta = tf.linspace(0.0, 180.0, tf.shape(radon_image)[1])
    
    angles_count = tf.shape(theta)[0]
    
    if angles_count.eval() != tf.shape(radon_image)[1].eval():
        raise ValueError("The given ``theta`` does not match the number of projections in ``radon_image``.")
    
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
  
    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError("Unknown filter: %s" % filter)
  
    img_shape = tf.shape(radon_image)[0].eval()
    if output_size is None:
    # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(math.floor(math.sqrt((img_shape) ** 2 / 2.0)))
    print(output_size)
    
# In[3]:

