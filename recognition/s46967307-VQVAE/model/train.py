from modules import AE
from dataset import load_data

# Initialize the Model, Print a summary.
model = AE()
model.compile(optimizer='adam')
print(model.encoder.summary())
print(model.decoder.summary())

# Load the Data.
data = load_data()
print(data)

# Begin model training and validation.

# Save model.
