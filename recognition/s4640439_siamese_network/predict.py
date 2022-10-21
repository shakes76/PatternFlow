import tensorflow as tf
from train import *
from dataset import *

TEST_DATA_POSITIVE_LOC = "ad_test"
TEST_DATA_NEGATIVE_LOC = "nc_test"

def main():
    """
    Used to load pre-trained models and then evaluate them using previously unseen test data
    """

    # load testing data
    test_data_positive = load_data(AD_TRAIN_PATH, TEST_DATA_POSITIVE_LOC)
    test_data_negative = load_data(NC_TRAIN_PATH, TEST_DATA_NEGATIVE_LOC)

    # load models
    siamese_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, "siamese_model.h5"))
    binary_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, "binary_model.h5"))

    # generate labels - 1: positive, 0: negative
    pos_labels = np.ones(test_data_positive.shape[0])
    neg_labels = np.zeros(test_data_negative.shape[0])

    # convert image data to embeddings
    pos_embeddings = siamese_model.predict(test_data_positive)
    neg_embeddings = siamese_model.predict(test_data_negative)

    # merge positive and negative datasets
    embeddings = np.concatenate((pos_embeddings, neg_embeddings))
    labels = np.concatenate((pos_labels, neg_labels))

    results = binary_model.evaluate(embeddings, labels)

    print(results)

if __name__ == "__main__":
    main()