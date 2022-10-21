"""
Showing example usage of your trained model
Print out any results and / or provide visualisations where applicable
"""
import tensorflow as tf
from train import *
from dataset import *

"""
TODO:
Load saved binary classification model from train.py
Load test data using dataset.py
Test the data and print/plot results
"""

TEST_DATA_POSITIVE_LOC = "ad_test"
TEST_DATA_NEGATIVE_LOC = "nc_test"

def main():
    # load testing data
    test_data_positive = load_data(AD_TRAIN_PATH, TEST_DATA_POSITIVE_LOC)
    test_data_negative = load_data(NC_TRAIN_PATH, TEST_DATA_NEGATIVE_LOC)

    # load models
    siamese_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, "siamese_model.h5"))
    binary_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_DIR, "binary_model.h5"))

    

    results = binary_model.evaluate()

if __name__ == "__main__":
    main()