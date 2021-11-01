Brief introduction on the perceiver transformer architecture

Perceiver transformer architecture:

Introduction: Usual transformers have the problem of quadratic bottle-neck(output query vector has too much of a dimension to handle). In the perceiver transformer architecture, high dimensional input such as images, video is represented as byte array, where it uses cross-attention with the use of Latent array to solve this problem (The input sequence gets transformed to a shorter sequence through the Latent state). 

To be more specific, in each layer of the perceiver transformer architecture, the inputs are fed into the each of the cross attention layer with the use of the Latent array to perform cross-attention. This allows for a deeper neural net when compared to usual transformer model (one of the biggest improvement over past transformer models). Visually, this is represented in the image Perceiver_architecture.png in the images folder.

Moreover, in the perceiver transformer architecture, the transformers are invariant to positional encoding of the data, i.e. different permutation of the input data does not affect the overall model. This is accomplished through the Fourier encoding as described in the paper. 

As such, the perceiver transformer model consist of the following:
-	Cross attention layer (to reduce dimension of input size)
-	Latent transformer layer after applying cross attention with the input byte array (usual transformer layer for handling data after cross attention)
-	Fourier encoding (for invariance to positional encoding of data)

The specific detail for the cross attention and latent transformer layer are described in Cross_attention_layer.png and transformer_layer.png in the images foler,

Training process and outputs:

For this task, we use the Perceiver transformer model to classify laterality(left or right) of the OAI AKOA knee data.
To start the algorithm and inspect the training and evaluation process, run the preprocess_data.py file, with the IMG_DIR representing the directory of the input knee data. 
The training process (training and validation accuracy/loss) and test set accuracy of the Perceiver model are shown in training_process.png, with the plot 
shown in Accuracy_loss_graph.png.

Sample predictions of the model (after the model has been training) are shown in the image prediction_sample.png.

Splitting of training and test set data:

For this project, 80% of the data are used for training, 10% are used for validation and the remaining 10% are used for test set which can be seen in the commenting of
random_split() function in preprocess_data.py. I think this allows fair balance of data for training, validation and testing.