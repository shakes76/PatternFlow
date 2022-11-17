"""
predict.py showing example usage of my trained model.
"""

# All needed library for usage of the trained model
from keras import models
from sklearn.metrics import classification_report
from modules import GraphConvolution
from dataset import *

"""
Load the pretrained model. The pretrained model will be uploaded to GitHub repository so user don't need to
train the model again in the train.py and can make use of '4_layers_GCN_Model.h5' as path for prediction and 
inference. If user trains the model again in the train.py, the trained model will still be saved as 
'4_layers_GCN_Model.h5' file so user don't need to change the path of load_model() function. Note: the trained
model will be in the same directory as other source files.
"""
trained_GCN_model = models.load_model('4_layers_GCN_Model.h5',
                                      custom_objects={'GraphConvolution': GraphConvolution})
trained_GCN_model.summary()

"""
Load the data for inference or prediction. User can change other data set for prediction and inference if the 
format of their data set follow https://snap.stanford.edu/data/facebook-large-page-page-network.html
"""
(adjacency_matrix, features_matrix, targets, train_target, validation_target, test_target, train_data_mask,
 validation_data_mask, test_data_mask) = load_facebook_page_data('./facebook.npz')

"""
Making use of pretrained GCN model for prediction and inference
"""
result = trained_GCN_model.predict([features_matrix, adjacency_matrix], batch_size=features_matrix.shape[0])
print("Inference of classes for the first 10 pages:")
print("=========================================================")
for page in range(10):
    print("The real class of page", page, "is class", (np.argmax(targets, axis=1)[page]))
    print("The inferred class of page", page, "is class", (np.argmax(result, axis=1)[page]))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Prediction result with the whole data set: ")
print(classification_report(np.argmax(targets, axis=1), np.argmax(result, axis=1)))

