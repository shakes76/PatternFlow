import torch
from skimage import io
from isics_data_setup import *

class YOLO():
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.build_model()

    def build_model(self):
        # Medium size model is used given the simplicity of dataset
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    # Custom training loop I've come up with... so far unsure how to actually use this with a model like YOLO
    def train(self):
        train_data = None # ...
        self.print_info()
        lbfgs = torch.optim.LBFGS()
        loss_func = torch.nn.MSELoss()
        runningTrainLoss = []
        runningTestLoss = []
        runningTestAccuracy = []
        runningTestF1 = []
        def closure(X, y): # required for L-BFGS
            lbfgs.zero_grad()
            y_pred = self.model(X)
            loss = loss_func(y_pred, y)
            loss.backward()
            return loss
        for i in range(self.epochs):
            trainLoss = 0
            batchCount = 0
            self.model.train()
            for (X, y) in train_data:
                X = X.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.LongTensor)
                y = torch.nn.functional.one_hot(y, num_classes=2).type(torch.FloatTensor) # change to torch.cuda for slurm
                lbfgs.zero_grad()
                y_pred = self.model(X)
                loss = loss_func(y_pred, y)
                loss.backward()
                if closure:
                    optim_closure = lambda: closure(X, y)
                    lbfgs.step(optim_closure)
                else:
                    lbfgs.step()
                batchCount += 1
                print(f'{i+1} iterations complete')
                print(f'Loss: {loss}')
            print("Epoch complete")
        return runningTrainLoss, runningTestLoss, runningTestAccuracy, runningTestF1

    def predict(self):
        pass
    
    def display_sample(self):
        image = io.imread("sample_image.jpg")
        results = self.model(image)
        results.show()

    def print_info(self):
        print(f"PyTorch Version: {torch.__version__}")