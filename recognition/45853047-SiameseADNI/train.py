from json import load


from modules import siamese
from modules import classification_model
from dataset import load_siamese_data
from dataset import load_classify_data



def train():
    model = siamese(128, 128)

    siamese_train, siamese_val = load_siamese_data()
    classify_train, classify_val = load_classify_data()

    siamese_train = siamese_train.batch(32)
    siamese_val = siamese_val.batch(32)

    classify_train = classify_train.batch(32)
    classify_val = classify_val.batch(32)

    model.fit(siamese_train, epochs=30, validation_data=siamese_val)

    predict(siamese_val, model)

    # Build classification model using traing subnet
    subnet = model.get_layer(name="subnet")
    classifier = classification_model(subnet)

    classifier.fit(classify_train, epochs=10, validation_data=classify_val)

    predict(classify_val, classifier)
    

def predict(ds, model):
    for pair, label in ds:
        pred = model.predict(pair)
        for i in range(len(pred)):
            print(pred[i], label[i])
        break 

train()