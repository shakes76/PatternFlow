#modules.py

#Containing the source code of the components of your model.
#Each component must be implementated as a class or a function.


import torch
import torch.nn as nn
import torch.nn.functional as F


#######################################################
#                  3D ResNet
####################################################### 


class Residual_Identity_Block_R3D(nn.Module):
    def __init__(self, c_in, c_out,kernel_size, padding,stride):
        super(Residual_Identity_Block_R3D, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm3d(c_in),
                            nn.ReLU())
        self.branch     = nn.Sequential(
                            nn.Conv3d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding), 
                            nn.BatchNorm3d(c_out),
                            nn.ReLU(),
                            nn.Conv3d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))       
    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+x

        return x

class Residual_Conv_Block_R3D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding,stride):
        super(Residual_Conv_Block_R3D, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm3d(c_in),
                            nn.ReLU(),                            
                            )
        self.branch = nn.Sequential(
                            nn.Conv3d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding), 
                            nn.BatchNorm3d(c_out),
                            nn.ReLU(),
                            nn.Conv3d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))
        self.conv       = nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+self.conv(x)

        return x

class ResNet_3D(nn.Module):
    def __init__(self, identity_block, conv_block):
        super().__init__()
        
        self.prep = nn.Sequential(nn.Conv3d(1, 64, kernel_size=7, stride=2, dilation=1), 
                                  nn.BatchNorm3d(64), 
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool3d(3,stride=2)
                                  )

        self.block0_1 = self._make_residual_block_(identity_block, 64, 64  ,3,1,(0,0,0))
        self.block1_1 = self._make_residual_block_(identity_block, 64, 64  ,3,1,(0,0,0))

        self.block0_2 = self._make_residual_block_(conv_block,     64, 128 ,3,1,(2,2,2))
        self.block1_2 = self._make_residual_block_(identity_block, 128, 128 ,3,1,(0,0,0))

        self.block0_3 = self._make_residual_block_(conv_block,     128, 256 ,3,1,(2,2,2))
        self.block1_3 = self._make_residual_block_(identity_block, 256, 256 ,3,1,(0,0,0))

        self.block0_4 = self._make_residual_block_(conv_block,     256, 512 ,3,1,(2,1,1))
        self.block1_4 = self._make_residual_block_(identity_block, 512, 512 ,3,1,(0,0,0))

        self.lin_layer = nn.Sequential(     
                                            nn.AvgPool3d((1,5,5),stride=2),
                                            #nn.AdaptiveAvgPool2d((1,1)),
                                            nn.Dropout(p=0.1),
                                            nn.ReLU(),
                                            nn.Flatten(),
                                            #nn.Linear(12800, 4096),
                                            #nn.ReLU(),
                                            #nn.Linear(4096,1028),
                                            #nn.ReLU(),
                                            #nn.Linear(1028,128),
                                            nn.Linear(12800, 2*2048),

                                            nn.ReLU())
                                            #,
                                            #nn.Sigmoid())
        
        self.final = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())


    def _make_residual_block_(self, block, c_in, c_out,kernel_size,padding,stride):
        layers = []
        layers.append(block(c_in,c_out,kernel_size,padding,stride))

        return nn.Sequential(*layers)  


    #Print OutputShape during net
    def forward_once_print(self, x):   
        #Preperation
        print("Inital",x.shape)
        out = self.prep(x)
        print("Prep",out.shape)
        #layer1
        out = self.block0_1(out) 
        print("block0_1",out.shape)
        out = self.block1_1(out) 
        print("block1_1",out.shape)

        #layer2
        out = self.block0_2(out) 
        print("block0_2",out.shape)
        out = self.block1_2(out)  
        print("block1_2",out.shape)

        #layer3
        out = self.block0_3(out) 
        print("block0_3",out.shape)
        out = self.block1_3(out) 
        print("block1_3",out.shape)
        
        #layer4
        out = self.block0_4(out) 
        print("block0_4",out.shape)
        out = self.block1_4(out)
        print("block1_4",out.shape)

        #final
        #out = torch.squeeze(out, 2)
        print("squeeze",out.shape)

        out = self.lin_layer(out)
        print("lin_layer=output",out.shape)

        return out

    def forward_once(self, x):
        out = self.prep(x)

        #layer1
        out = self.block0_1(out) 
        out = self.block1_1(out)
        out = self.block1_1(out) 
        out = self.block1_1(out)

        #layer2
        out = self.block0_2(out) 
        out = self.block1_2(out)
        out = self.block1_2(out)
        out = self.block1_2(out)

        #layer3
        out = self.block0_3(out) 
        out = self.block1_3(out)
        out = self.block1_3(out)
        out = self.block1_3(out)


        #layer4
        out = self.block0_4(out) 
        out = self.block1_4(out)
        out = self.block1_4(out)
        out = self.block1_4(out)

        #final
        #out = torch.squeeze(out, 2)
        out = self.lin_layer(out)

        return out

    def forward(self, x,y,z):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        out_z = self.forward_once(z)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out).squeeze(1) 

        return out_x, out_y, out_z








#######################################################
#                  Classifier Net
#######################################################

class Net_clas3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.final = nn.Sequential(

            nn.Flatten(),
            nn.Linear(2*2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            #nn.Linear(2048, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 1),
            #nn.sigmoid();
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        #out_AD = torch.abs(img-img_AD)
        #out_NC = torch.abs(img-img_NC)
        
        #output = torch.cat((out_AD, out_NC), 1)
        output = self.final(img)
        output = self.sigmoid(output)
        
        return output



#######################################################
#                  Weight Initialisation
#######################################################


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
        
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)


#######################################################
#                  Loss Function
#######################################################
class TripletLoss(torch.nn.Module):

    def __init__(self, margin=128.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, output_3):
        m = nn.Sigmoid()
        distance_AP = F.pairwise_distance(output_1, output_2, keepdim = True, p=1.0, eps=1e-06)
        distance_AN = F.pairwise_distance(output_1, output_3, keepdim = True, p=1.0, eps=1e-06)
        #print(distance_AP[0].item(),distance_AN[0].item())
        loss_triplet = torch.mean(torch.clamp(distance_AP-distance_AN+self.margin,min=0))

        return loss_triplet

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

        #print(euclidean_distance)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        #loss_contrastive=torch.mean((1-label) * torch.pow(euclidean_distance, -2) + (label) * torch.pow((euclidean_distance), 2))
        #loss_contrastive=torch.mean(label*torch.clamp((torch.exp(20*euclidean_distance)-1),max=10000)+(1-label)*torch.clamp((torch.exp(1/euclidean_distance-1)-1),max=10000))
        #loss_contrastive=torch.mean(label*torch.clamp((torch.exp(euclidean_distance)-1),max=1)+(1-label)*torch.clamp((torch.exp(1/euclidean_distance-1)-1),max=1))
        
        #print("DISTANCE: ",torch.mean(euclidean_distance).item(),"- LOSS: ",loss_contrastive.item(),"- Label average", torch.mean(label).item(), flush=True)

        return loss_contrastive




    

    
