import tensorflow as tf

#https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12

def radon(image, theta = None, circle = True):
    if image.rank != 2:
        raise ValueError('The input image must be 2D')
    if theta is None:
        pass
    
    if circle:
        pass
    else:
        pass