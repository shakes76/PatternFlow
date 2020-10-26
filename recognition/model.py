from tensorflow.keras import layers, models, optimizers, preprocessing
from sklearn.metrics import classification_report

class KneeModel:
	def __init__(self, train_path, val_path, test_path):
		# initial the para of the model
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.image_height = 224
		self.image_width = 224
		self.batch_size = 32
		self.model = self.constructModel()

	def constructModel(self):
		# construct the model by CNN
		input_image = layers.Input(shape=(self.image_height, self.image_width, 3))

		conv1 = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same")(input_image)
		bn1 = layers.BatchNormalization(axis=3)(conv1)
		relu1 = layers.LeakyReLU(alpha=0.3)(bn1)
		pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu1)

		conv2 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool1)
		bn2 = layers.BatchNormalization(axis=3)(conv2)
		relu2 = layers.LeakyReLU(alpha=0.3)(bn2)
		pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu2)

		conv3 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool2)
		bn3 = layers.BatchNormalization(axis=3)(conv3)
		relu3 = layers.LeakyReLU(alpha=0.3)(bn3)
		pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu3)

		conv4 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(pool3)
		bn4 = layers.BatchNormalization(axis=3)(conv4)
		relu4 = layers.LeakyReLU(alpha=0.3)(bn4)
		pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu4)

		flatten = layers.Flatten()(pool4)
		full = layers.Dropout(0.5)(layers.Dense(64, activation='relu')(flatten))
		output = layers.Dense(2, activation='sigmoid')(full)

		model_ = models.Model(input_image, output)
		model_.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
		model_.summary()
		return model_



