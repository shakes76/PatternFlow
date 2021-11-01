from model import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report

def main():
    """
    Driver function to execute example found in GCN Model.ipynb
    """
    ## PREPROCESSING ##

    # load data
    X_features, labels, edges = load_data()

    # check loaded properly
    num_classes = len(np.unique(labels))
    num_nodes = X_features.shape[0]
    num_features = X_features.shape[1]
    num_edges = len(edges)/2

    # adjacency matrix
    A = get_adj_matrix(labels, edges)

    # normalise
    A = normalise_adj(A)

    # Get indices for splitting set
    train_idx, val_idx, test_idx = split_index(labels)

    # Apply mask
    train_mask = np.zeros((num_nodes,), dtype = bool)
    val_mask = np.zeros((num_nodes,), dtype = bool)
    test_mask = np.zeros((num_nodes,), dtype = bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # One-hot encoding
    encoded_labels, classes, encoder = encode(labels)

    ## BUILD MODEL ##

    # Parameters
    channels = 16 #num for first layer
    dropout = 0.5 #rate
    l_rate = 1e-2 #learning rate
    l2_reg = 2.5e-4 # regularisation rate
    epochs = 200 #number of epochs

    # Create and Compile
    model = GCN_Model(num_features, num_classes, channels, dropout, l2(l2_reg), num_nodes)
    model.compile(optimizer = Adam(learning_rate = l_rate), loss = 'categorical_crossentropy', metrics = ['acc'])

    ## TRAINING ##
    validation_data = ([X_features, A], encoded_labels, val_mask)
        
    history = model.fit([X_features, A], encoded_labels, sample_weight = train_mask, 
        epochs = epochs, batch_size = num_nodes, validation_data = validation_data, shuffle = False)

    # Predict
    y_predictions = model.predict([X_features, A], batch_size = num_nodes)
    y_true = np.argmax(encoded_labels[test_mask], axis = 1)
    y_pred = np.argmax(y_predictions[test_mask], axis = 1)

    report = classification_report(y_true,y_pred)

    ## PLOT Using TSNE  ##
    plot_tsne(y_predictions, encoded_labels, num_classes)

if __name__ == "__main__":
    main()