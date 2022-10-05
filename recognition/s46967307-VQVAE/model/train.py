from modules import AE

# Initialize the Model, Print a summary.
model = AE()
model.compile(optimizer='adam')
print(model.encoder.summary())
print(model.decoder.summary())

# Load the Data.

# Begin model training and validation.

# Save model.
