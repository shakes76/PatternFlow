import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from iunet import build_iunet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

image_generator = ImageDataGenerator()

training_gen = image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1-2_Training_Input_x2"],
    target_size=(512, 384),
    color_mode="grayscale",
    class_mode='input')

training_gt_gen = image_generator.flow_from_directory(
    directory=os.path.expanduser('~'),
    batch_size=2594,
    shuffle=False,
    classes=["ISIC2018_Task1_Training_GroundTruth_x2"],
    target_size=(512, 384),
    color_mode="grayscale",
    class_mode='input')

x_train = training_gen.next()[0]
y_train = training_gt_gen.next()[0]

data_gen = image_generator.flow(x=x_train, y=y_train)

model = build_iunet()
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=CategoricalCrossentropy())
model.fit(x=data_gen, epochs=50)

model.save("model.tf")
