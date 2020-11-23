import tensorflow as tf
def create_model():
    covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    covn_base.trainable = True
    # Freeze the front layer and train the last seven layers
    for layers in covn_base.layers[:-7]:
        layers.trainable = False
    # Building models
    model = tf.keras.Sequential()
    model.add(covn_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D())  # Adding global average pooling layer
    model.add(tf.keras.layers.Dense(512, activation='relu'))  # Add full connectivity layer
    model.add(tf.keras.layers.Dropout(rate=0.5))  # Add dropout layer to prevent over fitting
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Add output layer(2 categories)
    model.summary()  # Print parameter information of each layer

    # Compiling model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  # Using Adam optimizer, the learning rate is 0.001
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Cross entropy loss function
                  metrics=["accuracy"])  # evaluation function
    return model

img_path = '/home/lbd855/DeepLearning/test/'

model = create_model()
model.load_weights('/home/lbd855/DeepLearning/save_weights/myweights.h5')

R_label_list = ['RIGHT', 'Right', 'R_I_G_H_T']
L_label_list = ['LEFT', 'Left', 'L_E_F_T']

test_images = [img_path + i for i in os.listdir(img_path)]
sum = len(test_images)
num = 0
for i, image_file in enumerate(test_images):
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.densenet.preprocess_input(x)
    preds = model.predict(x, verbose=0)
    print(preds)
    if preds[0, 0] > preds[0, 1]:
        if any(label in image_file for label in L_label_list):
            num = num + 1
    else:
        if any(label in image_file for label in R_label_list):
            num = num + 1
acc = num / sum
print("Accuracy: {} ".format(acc))
