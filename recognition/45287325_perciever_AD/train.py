import torch
import torch.optim as optim
import torch.nn as nn
from dataset import train_data_loader

from modules import Perciever

DATA_PATH = "./Images/AD_NC"

# Training parameters
EPOCHS = 10
BATCH_SIZE = 5
L_R = 0.005

trainloader = train_data_loader(DATA_PATH, BATCH_SIZE)

# Model Parameters
NUM_LATENTS = 50
DIM_LATENTS = 200
NUM_CROSS_ATTENDS = 1
DEPTH_LATENT_TRANSFORMER = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Perciever(NUM_LATENTS, DIM_LATENTS, DEPTH_LATENT_TRANSFORMER, NUM_CROSS_ATTENDS)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=L_R)

for epoch in range(EPOCHS):
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

        # print statistics
        running_loss += loss.item()

        if i % 50 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0

print('Finished Training')

MODEL_PATH = './perciever.pth'
torch.save(model.state_dict(), MODEL_PATH)