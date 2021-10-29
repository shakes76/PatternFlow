# -*- coding: utf-8 -*-
# Author: Tianjie Shi 
# Last update: 13/10/2019

import tensorflow as tf
tf.compat.v1.enable_eager_execution() # Make eager execution available

# Function1 -- unique_inverse
def unique_inverse(image):
    """
    Find the indices of the unique tensor that reconstruct the input image.

    Parameters
    ----------
    image : tensor
            Input a tesnor(image) which has already be flatten(1-D)

    Returns
    -------
    inv_idx: tensor
           The indices to reconstruct the original value from the
        unique value of image(tensor).

    Examples
    --------
    >>> A = [1,4,5,5,2,10,2,3,4,3,9]
    >>> A = tf.convert_to_tensor(A)
    >>> print(unique_inverse(A))
    tf.Tensor([0 3 4 4 1 6 1 2 3 2 5], shape=(11,), dtype=int32)
    """
    
    # convert data type to tf.int32 (tf.uint8(Most common) --> tf.int32)
    image = tf.cast(image,dtype = tf.int32)
    # Sort the values in tensor and return index
    perm = tf.argsort(image)
    # get the value of tensor
    perm = perm.numpy()
    image = image.numpy()
    #array operation
    aux = image[perm]
    #create a zero tensor
    mask = tf.zeros(aux.shape, dtype=tf.bool)
    mask = mask.numpy()
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    # dtype convert to tf.int32
    k = tf.cast(mask,dtype = tf.int32)
    # Compute the cumulative sum of the tensor
    imask = tf.cumsum(k) - 1
    inv_idx = tf.zeros(mask.shape, dtype=tf.int32)
    inv_idx = inv_idx.numpy()
    inv_idx[perm] = imask
    # reconvert to tensor to return 
    inv_idx = tf.convert_to_tensor(inv_idx)
    return inv_idx
   

# Function2 -- _interpolate
def _interpolate( dx_T, dy_T, x, name='interpolate'):
    """
    One-dimensional linear interpolation.(same as numpy.interp), but only can 
    interpolate one value each time

    Parameters
    ----------
    dx_T: 1-D sequence of floats
         The x-coordinates of the data points, 

    dy_T: 1-D sequence of floats
         The y-coordinates of the data points, same length as `dx_T`
    
    x: singe value array -- such as [3]...
    The x-coordinate at which to evaluate the interpolated value.

    Returns
    -------
    result: float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape is equal to  1.

    Examples
    --------
    >>> x = np.linspace(0, 2 * np.pi, 10)
    >>> y = np.sin(x)
    >>> o = [3]
    >>> interpolate(x,y,o)
    <tf.Tensor: id=432, shape=(), dtype=float64, numpy=0.13873468177796913>

    """
    
    with tf.compat.v1.variable_scope(name):
        
        #create a new variable
        with tf.compat.v1.variable_scope('neighbors'):
             
            delVals = tf.subtract(dx_T, x)
            ind_1   = tf.argmax(tf.sign( delVals ))
            ind_0   = ind_1 - 1

        with tf.compat.v1.variable_scope('calculation'):
            # get the value 
            value   = tf.cond( x[0] <= dx_T[0], 
                              lambda : dy_T[:1], 
                              lambda : tf.cond( 
                                     x[0] >= dx_T[-1], 
                                     lambda : dy_T[-1:],

                                     lambda : (dy_T[ind_0] +                
                                               (dy_T[ind_1] - dy_T[ind_0])  
                                               *(x-dx_T[ind_0])/            
                                               (dx_T[ind_1]-dx_T[ind_0]))
                             ))

        result = tf.multiply(value[0], 1, name='y')

    return result


# Function3 -- _match_cumulative_cdf
def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.

    Parameters
    ----------
    source: tensor
        input a source image 

    template: tensor 
        input a reference image 


    Returns
    ------- 
    result: tensor 
       a new tensor after cumulative matching which has same shape with source image

    """
    # flatten the images
    source_flatten = tf.reshape(source,[-1])
    template_flatten = tf.reshape(template,[-1])

    #data type convert to tf.int32
    source_flatten_sort = tf.cast(source_flatten ,dtype =tf.int32)
    #sort the value in tensor so to get the matched count value(src_counts)
    source_flatten_sort = tf.sort(source_flatten_sort)
    template_flatten = tf.cast(template_flatten ,dtype =tf.int32)
    template_flatten = tf.sort(template_flatten)
    # 
    src_values, src_unique_indices, src_counts = tf.unique_with_counts(source_flatten_sort)

    #get the unique indiex 
    src_indice = unique_inverse(source_flatten)
    #
    tmpl_values,tmpl_unique_indices,tmpl_counts = tf.unique_with_counts(template_flatten)
    tmpl_values = tf.cast(tmpl_values,dtype=tf.float64)
    # calculate normalized quantiles for each tensor
    source_size = tf.size(source_flatten)
    template_size = tf.size(template_flatten)
    src_quantiles = tf.cumsum(src_counts) / source_size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template_size
    #interpolate
    #Create a list to save the interpolate value
    interp_a_values = []
    for i in src_quantiles.numpy():
        interp_a_values.append(_interpolate(tmpl_quantiles, tmpl_values, tf.constant([i])))
    interp_a_values = tf.convert_to_tensor(interp_a_values).numpy()
    guodu = interp_a_values[src_indice]
    #convert_to_tensor
    guodu = tf.convert_to_tensor(guodu)
    result = tf.reshape(guodu,tf.shape(source))
    return result
    
def match_histograms(image, reference, multichannel=False):

    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : tensor
        Input image. Can be gray-scale or in color.
    reference : tensor
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : tensor
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    """   
    #Determine whether the number of channels in the two pictures is the same
    if tf.rank(image).numpy() != tf.rank(reference).numpy():
        raise ValueError('Image and reference must have the same number of channels.')
    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        #Create a list to save the results of each channel
        matched_channel = []
        
        for channel in range(image.shape[-1]):
            matched_channel.append(_match_cumulative_cdf(image[..., channel], reference[..., channel]))
        # Combine the results of all channels to convert to the picture.
        matched = tf.stack([matched_channel[0],matched_channel[1],matched_channel[2]], axis=2)
        #Because the value obtained is floating point,
        # it is converted to floating point type [0,1] to get a new picture.
        matched = matched/255.
    
    else:
        matched = _match_cumulative_cdf(image, reference)
        matched = matched/255.
    return matched
