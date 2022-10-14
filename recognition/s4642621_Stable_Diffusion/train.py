from imports import *
from modules import *
from dataset import *

plt.style.use('ggplot')

"""
Train the model, and regularly save and plot the respective losses
"""
def train(trainloader, valloader):
    model = UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 51

    loss_vals = []
    val_vals  = []

    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.0

        for index, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
            batch = batch.to(device)
            optimizer.zero_grad()

            # sample t uniformally for every image in the batch
            t = torch.randint(0, TIMESTEPS, (1,), device=device).long()

            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'  [{epoch+1}, {index + 1:5d}] loss: {running_loss / len(trainloader):.4f}', flush=True)
        loss_vals.append(running_loss / len(trainloader))
        val_vals.append(validate(model, valloader))

        # save model and update loss figure every 5 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model, "DiffusionModel" + str(epoch))

            # create loss graph
            fig, ax = plt.subplots()
            plt.plot(loss_vals, label="Training Loss")
            plt.plot(val_vals, label="Validation Loss")
            ax.set_ylabel('Loss')
            ax.set_title('Stable Diffusion Training/Validation Error')
            ax.set_xlabel('Epoch')
            ax.legend(loc="upper right")
            plt.show()
            plt.savefig("loss.png") 

"""
Calculate the validation loss
"""
def validate(model, valloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_vals = []
    running_loss = 0.0
    for index, batch in tqdm(enumerate(valloader), total=len(valloader)):
        batch = batch.to(device)
        # sample t uniformally for every image in the batch
        t = torch.randint(0, TIMESTEPS, (1,), device=device).long()
        loss = get_loss(model, batch, t)
        running_loss += loss.item()

    print(f'  [validation loss: {running_loss / len(valloader):.4f}', flush=True)
    val_vals.append([running_loss / len(valloader)])
    return val_vals

if __name__ == "__main__":
    print("Output-------------------------------------")
    print("Loading Dataset...", flush=True)
    total_dataloader = load_dataset(batch_size=16)
    val_dataloader   = load_dataset(batch_size=16, val_set=True)
    print("Loaded Dataset", flush=True)
    print("Training.....", flush=True)
    train(total_dataloader, val_dataloader)
    print("Finished Training", flush=True)
