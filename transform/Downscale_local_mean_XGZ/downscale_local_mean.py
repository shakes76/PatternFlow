import tensorflow as tf
print("TF Version: ", tf.__version__)
tf.InteractiveSession()

def downscale_local_mean(image, factors, cval=0, clip=True):
  
  return block_reduce(image, factors, tf.reduce_mean, cval)
