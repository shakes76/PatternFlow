import tensorflow as tf
    
def Contrast_Sigmoid(array,cutoff=0.5,gain=10.,inv=False):
    #convert the numpy array to tensor 
    tensor = tf.convert_to_tensor(array)
    tf.assert_non_negative(tensor) #ensure tensor values are all non negative
    tensor_dtype = tensor.dtype #save type so can return same type as input
    tensor = tf.cast(tensor, dtype=tf.float32)

    #First -> define scale, sclae is a 1x1 tensor
    scale = tf.cast(tf.reduce_max(tensor) - tf.reduce_min(tensor), dtype=tf.float32) #scale to scale image by (to 0->1)
    #Second -> Apply sigmoid function (checking for inverse (inv) parameter)
    if inv:
        out = (1 - 1 / (1. + tf.exp(gain * (cutoff - tensor / scale)))) * scale
    else:
        print(gain, cutoff, tensor, scale.eval())
        out = tf.divide(1, tf.cast(1 + tf.exp(gain * (cutoff - tf.divide(tensor, scale))),dtype=tf.float32 )) * scale
    out = tf.cast(out, dtype=tensor_dtype) #fixing return type
    return out
    
   
    
    # 
    
