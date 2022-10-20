import dataset
import torch
import vit_pytorch


test_data = dataset.torch_test('test')
model = torch.load('saved/best_model.pth').cuda()

test_iter = iter(test_data)
test_acc = 0
test_size = 0
model.eval()
for batch, labels in test_iter:
    batch, labels = batch.cuda(),labels.cuda()
    prediction = model(batch)
    acc = sum(torch.argmax(prediction,dim=1) == torch.argmax(labels,dim=1)).item()
    test_acc += acc
    test_size+= len(batch)

test_acc /= test_size

print(test_acc)