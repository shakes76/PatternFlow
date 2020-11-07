# Setup the optimizer and loss function
smooth=1
def dice_coef(trainGen,testGen, smooth=1):
  intersection = K.sum(trainGen* testGen, axis=[1,2,3])
  union = K.sum(trainGen, axis=[1,2,3]) + K.sum(testGen, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def dice_coef_loss(trainGen,testGen):
  return -dice_coef(trainGen, testGen)

model.compile(optimizer=Adam(lr=1e-4), loss=-dice_coef_loss, metrics=[dice_coef])

# Set the proper checkpoint callback to save model.
model_checkpoint = ModelCheckpoint(
    'best_model.hdf5', monitor='loss', verbose=1, save_best_only=True)

# Model training.
history = model.fit_generator(
    trainGen,
    steps_per_epoch=train_steps,
    epochs=epochs,
    callbacks=[model_checkpoint],
    validation_data=valGen,
    validation_steps=validate_steps,
)