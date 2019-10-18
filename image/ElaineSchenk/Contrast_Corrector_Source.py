import tensorflow as tf
    
def Contrast_Sigmoid(array,cutoff=0.5,gain=10.,inv=False):
    
    #First -> convert the numpy array to tensor 
    tensor = tf.convert_to_tensor(array)
    
    tf.assert_non_negative(tensor) #ensure tensor values are all non negative
    
    tensor_dtype = tensor.dtype #save type so can return same type as input
    
    tensor = tf.cast(tensor, dtype=tf.float32) #its really frustrating to do this but otherwise divison by integer if array is an int
    #Second-> define scale, scale is a 1x1 tensor
    
    scale = tf.cast(tf.reduce_max(tensor) - tf.reduce_min(tensor), dtype=tf.float32) #scale to scale image by (to 0->1)
    #Third -> Apply sigmoid function (checking for inverse (inv) parameter)
    
    if inv:
        out = (1 - tf.divide(1, tf.cast(1 + tf.exp(gain * (cutoff - tf.divide(tensor,scale))), dtype=tf.float32))) * scale
    else:
        #print(gain, cutoff, tensor, scale.eval()) #having issues with data types
        out = tf.divide(1, tf.cast(1 + tf.exp(gain * (cutoff - tf.divide(tensor, scale))),dtype=tf.float32 )) * scale
    
    out = tf.cast(out, dtype=tensor_dtype) #fixing return type
    
    return out
    
