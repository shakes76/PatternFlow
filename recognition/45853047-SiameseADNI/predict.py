from tensorflow.keras.models import load_model
from dataset import load_classify_data
import tensorflow as tf

CLASSIFIER_PATH = 'recognition\\45853047-SiameseADNI\\models\Classifier.h5'

def predict():
    """ 
    Use the classification model to make predictions
    """
    # Evaluate
    classifier = load_model(CLASSIFIER_PATH)
    classify_test_data = load_classify_data(testing=True)

    # Evaluate the classifier
    classifier.evaluate(classify_test_data)

    # Show predictions for one batch (32 predictions)
    for pair, label in classify_test_data:
        pred = classifier.predict(pair)
        for i in range(len(pred)):
            print("Prediction: ", pred[i])
            print("Actual: ", label[i])
        break 

predict()