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
                                  #nn.Linear(4096, 256))
                                  nn.Sigmoid())

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



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
        
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

