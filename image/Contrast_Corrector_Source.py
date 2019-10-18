import tensorflow as tf
    
def Contrast_Sigmoid(array,cutoff=0.5,gain=10,inv=false):
    #convert the numpy array to tensor 
    tensor = tf.convert_to_tensor(array)
    tf.assert_non_negative(tensor) #ensure tensor values are all non negative
    tensor_dtype = tensor_img.dtype #save type so can return same type as input
    
    #First -> define scale
    scale = tf.reduce_max(tensor) - tf.reduce_min(tensor) #scale to scale image by (to 0->1)
    #Second -> Apply sigmoid function (checking for inverse (inv) parameter)
    if inv:
        out = (1 - 1 / (1 + np.exp(gain * (cutoff - image / scale)))) * scale
    else:
        out = (1 / (1 + np.exp(gain * (cutoff - image / scale)))) * scale
    #out = #fixing return type
    return out
    
   
    
    # 
    