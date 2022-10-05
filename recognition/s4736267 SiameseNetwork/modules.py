#modules.py

#Containing the source code of the components of your model.
#Each component must be implementated as a class or a function.





import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 64, kernel_size=10, padding=0), 
                                  nn.ReLU(),                                                        
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, kernel_size=7, padding=0), 
                                  nn.ReLU(),                                  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(128, 128, kernel_size=4, padding=0), 
                                  nn.ReLU(),                                  
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(128, 256, kernel_size=4, padding=0), 
                                  nn.ReLU(),                                  
                                  nn.Flatten()                                                                   
                                  )
        self.lin_layer = nn.Sequential(nn.Linear(9216, 4096),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(4096, 256))
                                  #nn.Sigmoid())

        #self.pdist = nn.PairwiseDistance(p=1, keepdim=True)    
        #self.final = nn.Linear(4096, 1)

        #self.final = nn.Sequential(nn.Linear(4096, 1),
        #                           nn.Sigmoid())


    def forward_once(self, x):

        out = self.conv_layer(x)
        out = out.view(out.size()[0], -1)
        out = self.lin_layer(out)
        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out)

        return out_x, out_y

