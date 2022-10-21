'''
Metrics for UNet.
'''

from tensorflow.keras import backend

def dice_coe(y_ground, y_predicted):
    '''
    Dice coefficient metric of performance, equivalent to 2 X IoU.
    
    Args:
        y_ground (tf.Tensor): Ground truth segmentation masks.
        y_predicted (tf.Tensor): Predicted segmentation masks.
        
    Returns:
        float: Dice coefficient of y_ground and y_predicted.
    '''
    intersection = backend.sum(y_ground * y_predicted, axis=(1, 2))
    union = backend.sum(y_ground + y_predicted, axis=(1, 2))
    
    return 2 * (intersection) / (union)

def smoothed_jaccard_distance(y_ground, y_predicted, smoothing=10):
    '''
    Jaccard distance loss function with added smoothing for numerical stability; 
    smoothing level can act as a hyperparameter but is mainly effective only where 
    batch normalisation has not been implemented.
    
    Args:
        y_ground (tf.Tensor): Ground truth segmentation masks.
        y_predicted (tf.Tensor): Predicted segmentation masks.
        
    Returns:
        float: Smoothed Jaccard distance of y_ground and y_predicted.
    '''
    intersection = backend.sum(y_ground*y_predicted, axis=(1, 2))
    union = backend.sum(y_ground + y_predicted, axis=(1, 2))
    jaccard_index = (intersection + smoothing)/(union - intersection + smoothing)
    
    return (1 - jaccard_index)*smoothing

