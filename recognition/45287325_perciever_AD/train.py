import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import train_data_loader
from modules import Perciever

DATA_PATH = "./Images/AD_NC"

# Training parameters
EPOCHS = 10
BATCH_SIZE = 5
L_R = 0.005

trainloader = train_data_loader(DATA_PATH, BATCH_SIZE)

# Model Parameters
NUM_LATENTS = 32
DIM_LATENTS = 128
NUM_CROSS_ATTENDS = 1
DEPTH_LATENT_TRANSFORMER = 4

# Path to save the model to
MODEL_PATH = './perciever.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Perciever(NUM_LATENTS, DIM_LATENTS, DEPTH_LATENT_TRANSFORMER, NUM_CROSS_ATTENDS)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=L_R)

loss_data = []
accuracy = []

for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
    print(f"Epoch {epoch} completed")
    loss_data.append(running_loss / 500)
    accuracy.append(correct / total * 100)

plt.plot(loss_data)
plt.xlabel('EPOCH')
plt.ylabel('Average Loss')
plt.show()   

plt.plot(accuracy)
plt.xlabel('EPOCH')
plt.ylabel('Training Accuracy')
plt.show()  

torch.save(model.state_dict(), MODEL_PATH)