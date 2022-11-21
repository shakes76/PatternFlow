import dataset
import torch
import modules

#Rebuild Model, drop other components as training is not required.
model, *_ = modules.build_model(
    dim=512, 
    image_size=256, 
    patch_size=32,
    num_classes=2,
    depth=8,
    heads=12,
    mlp_dim=1024,
    channels=1,
    dropout=0.5,
    emb_dropout=0.5,
    lr = 0.0001
)
model.cuda()



test_data = dataset.torch_test('test') #Load test data

model.load_state_dict(torch.load('saved/best_model.pth')) #Load weights from file

test_iter = iter(test_data) #Create generator
test_acc = 0
test_size = 0
model.eval()

#Parse through dataset
for batch, labels in test_iter:
    batch, labels = batch.cuda(),labels.cuda()
    prediction = model(batch)
    acc = sum(torch.argmax(prediction,dim=1) == torch.argmax(labels,dim=1)).item()
    test_acc += acc
    test_size+= len(batch)

test_acc /= test_size

print(test_acc) #Report test accuracy