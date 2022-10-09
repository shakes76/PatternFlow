from json import load


from modules import siamese
from modules import classification_model
from dataset import load_siamese_data
from dataset import load_classify_data



def train():
    # Siamese model
    siamese = siamese(128, 128)

    # Siamese model data
    siamese_train, siamese_val = load_siamese_data()
    siamese_train = siamese_train.batch(32)
    siamese_val = siamese_val.batch(32)

    # Classification model data
    classify_train, classify_val = load_classify_data(testing=False)
    classify_train = classify_train.batch(32)
    classify_val = classify_val.batch(32)

    # Test Data
    test = load_classify_data(testing=True)
    test = test.batch(32)

    siamese.fit(siamese_train, epochs=30, validation_data=siamese_val)

    predict(siamese_val, siamese)

    # Build classification model using trained subnet
    classifier = classification_model(siamese.get_layer(name="subnet"))

    classifier.fit(classify_train, epochs=10, validation_data=classify_val)

    # TODO: move this to predict.py
    # see predictions
    predict(test, classifier)

    # Evaluate model
    classifier.evaluate(test)

    
    
# TODO: move to predict.py
def predict(ds, model):
    for pair, label in ds:
        pred = model.predict(pair)
        for i in range(len(pred)):
            print(pred[i], label[i])
        break 

train()