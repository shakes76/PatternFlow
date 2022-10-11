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
                                  #nn.Sigmoid(),
                                  #nn.Linear(9216, 4096),
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

class Net_batchnom(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 64, kernel_size=10, padding=0), 
                                  nn.ReLU(), 
                                  nn.Conv2d(64, 64, kernel_size=7, padding='same'), 
                                  nn.ReLU(),                                                        
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(64),

                                  nn.Conv2d(64, 128, kernel_size=7, padding=0), 
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=7, padding='same'), 
                                  nn.ReLU(),                                   
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(128),

                                  nn.Conv2d(128, 128, kernel_size=4, padding=0), 
                                  nn.ReLU(),   
                                  nn.Conv2d(128, 128, kernel_size=4, padding='same'), 
                                  nn.ReLU(),                               
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(128),

                                  nn.Conv2d(128, 256, kernel_size=4, padding=0), 
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, kernel_size=4, padding='same'), 
                                  nn.ReLU(),                                  
                                  nn.BatchNorm2d(256),                                  
                                  nn.Flatten()                                                                   
                                  )
        self.lin_layer = nn.Sequential(nn.Linear(9216, 4096),
                                  #nn.ReLU(inplace=True),
                                  #nn.Linear(4096, 256),
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

#######################################################
#                  ResNET
#######################################################



class Residual_Identity_Block(nn.Module):
    def __init__(self, c_in, c_out,kernel_size, padding):
        super(Residual_Identity_Block, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm2d(c_in),
                            nn.ReLU())
        self.branch     = nn.Sequential(
                            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding), 
                            nn.BatchNorm2d(c_out),
                            nn.ReLU(),
                            nn.Conv2d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))       
    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+x

        return x

class Residual_Conv_Block(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding):
        super(Residual_Conv_Block, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm2d(c_in),
                            nn.ReLU(),                            
                            )
        self.branch = nn.Sequential(
                            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=2, padding=padding), 
                            nn.BatchNorm2d(c_out),
                            nn.ReLU(),
                            nn.Conv2d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))
        self.conv       = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+self.conv(x)

        return x

class ResNet18(nn.Module):
    def __init__(self, identity_block, conv_block):
        super().__init__()
        
        self.prep = nn.Sequential(nn.Conv2d(20, 64, kernel_size=10, padding=0), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2)
                                  )

        self.block0_1 = self._make_residual_block_(identity_block, 64, 64,3,1)
        self.block1_1 = self._make_residual_block_(identity_block, 64, 64,3,1)

        self.block0_2 = self._make_residual_block_(conv_block, 64, 128,3,1)
        self.block1_2 = self._make_residual_block_(identity_block, 128, 128,3,1)

        self.block0_3 = self._make_residual_block_(conv_block, 128, 128,3,1)
        self.block1_3 = self._make_residual_block_(identity_block, 128, 128,3,1)

        self.block0_4 = self._make_residual_block_(conv_block, 128, 256,3,1)
        self.block1_4 = self._make_residual_block_(identity_block, 256, 256,3,1)

        self.lin_layer = nn.Sequential(nn.Linear(9216, 4096),nn.Relu(),nn.Linear(4096,128),nn.Sigmoid())

    def _make_residual_block_(self, block, c_in, c_out,kernel_size,padding):
        layers = []
        layers.append(block(c_in,c_out,kernel_size,padding))

        return nn.Sequential(*layers)  



    def forward_once_print(self, x):
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
#layer1
        out = self.block0_3(out) 
        print("block0_3",out.shape)
        out = self.block1_3(out) 
        print("block1_3",out.shape)
#layer2
        out = self.block0_4(out) 
        print("block0_4",out.shape)
        out = self.block1_4(out)
        print("block1_4",out.shape)

        out = out.view(out.size()[0], -1)  ##=> euqals flatten
        print("view",out.shape)
        out = self.lin_layer(out)
        print("lin_layer=output",out.shape)

        return out

    def forward_once(self, x):
        out = self.prep(x)
        #layer1
        out = self.block0_1(out) 
        out = self.block1_1(out) 
        #layer2
        out = self.block0_2(out) 
        out = self.block1_2(out)
        #layer3
        out = self.block0_3(out) 
        out = self.block1_3(out)
        #layer4
        out = self.block0_4(out) 
        out = self.block1_4(out)

        out = out.view(out.size()[0], -1)
        out = self.lin_layer(out)

        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out)

        return out_x, out_y


class Net_3D(nn.Module):
    def __init__(self):
        super().__init__()
        
#https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/ oder lyer conv, batch, activation, pool

        self.conv_layer = nn.Sequential(
                                  #=> 1x20x210x210
                                  nn.Conv3d(1, 16, kernel_size=3, padding='same',stride=1), 
                                  nn.BatchNorm3d(16),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  #nn.Conv3d(16, 16, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(16),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  ##nn.Mish(), 
                                  #=> 16x10x105x105
                                  #nn.Conv3d(16, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(),  
                                  #nn.Conv3d(32, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  ##=> 32x5x52x52
                                  #nn.Conv3d(32,32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(),  
                                  #nn.Conv3d(32,32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  ##=> 32x2x26x26   
                                  #nn.Conv3d(32, 32, kernel_size=1, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),  
                                  #nn.Mish(),  
                                  #nn.Conv3d(32, 32, kernel_size=1, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),  
                                  #nn.Mish(),                                                                                                                     
                                  ##=> 32x1x13x13

                                  )
        self.lin_layer = nn.Sequential(nn.Linear(13*13*32, 1024),
                                  #nn.Mish(),
                                  #nn.Linear(4096, 256),
                                  nn.Sigmoid())

        #self.pdist = nn.PairwiseDistance(p=1, keepdim=True)    
        #self.final = nn.Linear(4096, 1)

        #self.final = nn.Sequential(nn.Linear(4096, 1),
        #                           nn.Sigmoid())


    def forward_once(self, x):
        
        out = self.conv_layer(x)
        print("afterconv",out.shape)
        out = out.view(out.size()[0], -1)  ##=> euqals flatten
        out = self.lin_layer(out)
        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out)

        return out_x, out_y

class Net_3D_BCE(nn.Module):
    def __init__(self):
        super().__init__()
        
#https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/ oder lyer conv, batch, activation, pool

        self.conv_layer = nn.Sequential(
                                  #=> 1x20x210x210
                                  nn.Conv3d(1, 16, kernel_size=3, padding='same',stride=1), 
                                  nn.BatchNorm3d(16),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  nn.Mish(), 
                                  #=> 16x10x105x105
                                  nn.Conv3d(16, 32, kernel_size=3, padding='same',stride=1), 
                                  nn.BatchNorm3d(32),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  nn.Mish(),  
                                  #=> 32x5x52x52
                                  nn.Conv3d(32,64, kernel_size=3, padding='same',stride=1), 
                                  nn.BatchNorm3d(64),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  nn.Mish(),  
                                  #=> 32x2x26x26   
                                  nn.Conv3d(64, 64, kernel_size=1, padding='same',stride=1), 
                                  nn.BatchNorm3d(64),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),  
                                  nn.Mish(),                                                                                                                       
                                  #=> 32x1x13x13

                                  )
        self.lin_layer = nn.Sequential(nn.Linear(2*13*13*32, 4096),
                                  #nn.Sigmoid(),
                                  #nn.Linear(4096, 256),
                                  nn.Sigmoid())

        #self.pdist = nn.PairwiseDistance(p=1, keepdim=True)    
        #self.final = nn.Linear(4096, 1)

        self.final = nn.Sequential(nn.Linear(4096, 1),
                                   nn.Sigmoid())


    def forward_once(self, x):
        
        out = self.conv_layer(x)
        #print(out.shape)
        out = out.view(out.size()[0], -1)  ##=> euqals flatten
        out = self.lin_layer(out)
        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        out  = torch.abs((out_x-out_y))
        out  = self.final(out).squeeze(1) 

        return out



class ResNet18_3D(nn.Module):
    def __init__(self, identity_block, conv_block):
        super().__init__()
        
        self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size=10, padding=0, stride=2), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU(inplace=True),
                                  #nn.MaxPool2d(2)
                                  )

        self.block0_1 = self._make_residual_block_(identity_block, 64, 64,7,3)
        self.block1_1 = self._make_residual_block_(identity_block, 64, 64,7,3)

        self.block0_2 = self._make_residual_block_(conv_block, 64, 128,7,3)
        self.block1_2 = self._make_residual_block_(identity_block, 128, 128,7,3)

        self.block0_3 = self._make_residual_block_(conv_block, 128, 128,5,2)
        self.block1_3 = self._make_residual_block_(identity_block, 128, 128,5,2)

        self.block0_4 = self._make_residual_block_(conv_block, 128, 256,3,1)
        self.block1_4 = self._make_residual_block_(identity_block, 256, 256,3,1)

        self.lin_layer = nn.Sequential(nn.Linear(9216, 4096),nn.Sigmoid())

    def _make_residual_block_(self, block, c_in, c_out,kernel_size,padding):
        layers = []
        layers.append(block(c_in,c_out,kernel_size,padding))

        return nn.Sequential(*layers)  

    def forward_once(self, x):
        out = self.prep(x)
        #layer1
        out = self.block0_1(out)
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

        out = out.view(out.size()[0], -1)
        out = self.lin_layer(out)

        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out)

        return out_x, out_y


#######################################################
#                  Classifier Net
#######################################################
class Net_clas3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.final = nn.Sequential(

            nn.BatchNorm1d(2*4096),
            nn.LeakyReLU(negative_slope=0.001),

            nn.Linear(2*4096, 2*2048),
            nn.BatchNorm1d(2*2048),
            nn.LeakyReLU(negative_slope=0.001),

            nn.Linear(2*2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.001),

            nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
            #nn.sigmoid();
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img,img_AD,img_NC):
        out_AD = torch.abs(img-img_AD)
        out_NC = torch.abs(img-img_NC)
        
        output = torch.cat((out_AD, out_NC), 1)
        output = self.final(output)
        output = self.sigmoid(output)
        
        return output



class Net_clas(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pdist = nn.PairwiseDistance(p=1, keepdim=False)
        #self.pdist = torch.cdist(p=1)    
        #self.final = nn.Linear(4096, 1)

        self.fc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),            
        )

        self.final = nn.Sequential(
            nn.Linear(2*256, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 1),
            #nn.sigmoid();
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img,img_AD,img_NC):
        out_AD = torch.abs(img-img_AD)
        out_NC = torch.abs(img-img_NC)
        
        out_AD = self.fc(out_AD)
        out_NC = self.fc(out_NC)

        output = torch.cat((out_AD, out_NC), 1)
        output = self.final(output)
        output = self.sigmoid(output)
        
        return output


#######################################################
#                  Siamese Newtork for BCEloss
#######################################################



class Net_binloss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 64, kernel_size=10, padding=0,stride=2), 
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
        self.lin_layer = nn.Sequential(nn.Linear(6400, 4096),
                                  #nn.ReLU(inplace=True),
                                  #nn.Linear(4096, 256))
                                  nn.Sigmoid())

        #self.pdist = nn.PairwiseDistance(p=1, keepdim=True)    
        #self.final = nn.Linear(4096, 1)

        self.final = nn.Sequential(nn.Linear(8192, 1),
                                  nn.Sigmoid())


    def forward_once(self, x):

        out = self.conv_layer(x)
        out = out.view(out.size()[0], -1)
        out = self.lin_layer(out)
        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        out = torch.cat((out_x, out_y), 1)
        out  = self.final(out)

        return out





def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
        
    if isinstance(m, nn.Conv3d):
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

        #print("modules.py euclidean_distance",euclidean_distance)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        #print("modules.py loss_contrastive",loss_contrastive)
        #print("moduley.py label", label)
        return loss_contrastive

#######################################

##############################
##
#
#############################

### RESNET 3D
def conv_block_R3D(in_channels, out_channels, pool=False):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm3d(out_channels), 
              nn.Mish()]
    return nn.Sequential(*layers)


class Residual_Identity_Block_R3D(nn.Module):
    def __init__(self, c_in, c_out,kernel_size, padding):
        super(Residual_Identity_Block_R3D, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm3d(c_in),
                            nn.Mish())
        self.branch     = nn.Sequential(
                            nn.Conv3d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding), 
                            nn.BatchNorm3d(c_out),
                            nn.Mish(),
                            nn.Conv3d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))       
    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+x

        return x

class Residual_Conv_Block_R3D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, padding):
        super(Residual_Conv_Block_R3D, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm3d(c_in),
                            nn.Mish(),                            
                            )
        self.branch     = nn.Sequential(
                            nn.Conv3d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding), 
                            nn.BatchNorm3d(c_out),
                            nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                            nn.Mish(),
                            nn.Conv3d(c_out, c_out,kernel_size=kernel_size, stride=1, padding=padding))
        self.conv       = nn.Sequential(
                            nn.Conv3d(c_in, c_out, kernel_size=2, stride=2, padding=padding-1))
                            #nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=padding-1), 
                            #nn.BatchNorm3d(c_out),
                            #nn.MaxPool3d(kernel_size=2,stride=2,padding=0))

    def forward(self, x):
        x = self.block_prep(x)
        #print("BRANCH:",self.branch(x).shape," - CONV:",self.conv(x).shape)
        x = self.branch(x)+self.conv(x)

        return x

                                  #nn.Conv3d(16, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(),  
                                  #nn.Conv3d(32, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ###nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 

#

                                  ##=> 1x20x210x210
                                  #nn.Conv3d(1, 16, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(16),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  #nn.Conv3d(16, 16, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(16),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  ##=> 16x10x105x105
                                  #nn.Conv3d(16, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(),  
                                  #nn.Conv3d(32, 32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  ##=> 32x5x52x52
                                  #nn.Conv3d(32,32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(),  
                                  #nn.Conv3d(32,32, kernel_size=3, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  #nn.Mish(), 
                                  ##=> 32x2x26x26   
                                  #nn.Conv3d(32, 32, kernel_size=1, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  #nn.MaxPool3d(kernel_size=2,stride=2,padding=0),  
                                  #nn.Mish(),  
                                  #nn.Conv3d(32, 32, kernel_size=1, padding='same',stride=1), 
                                  #nn.BatchNorm3d(32),
                                  ##nn.MaxPool3d(kernel_size=2,stride=2,padding=0),  
                                  #nn.Mish(),                                                                                                                     
                                  ##=> 32x1x13x13



class ResNet18_R3D(nn.Module):
    def __init__(self, identity_block, conv_block):
        super().__init__()     
                  
        self.prep = nn.Sequential(nn.Conv3d(1, 16, kernel_size=3, padding='same',stride=1), 
                                  nn.BatchNorm3d(16),
                                  nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
                                  nn.Mish())

        self.block0_1 = self._make_residual_block_(identity_block, 16, 16,3,1)
        self.block1_1 = self._make_residual_block_(identity_block, 16, 16,3,1)

        self.block0_2 = self._make_residual_block_(conv_block, 16, 32,3,1)
        self.block1_2 = self._make_residual_block_(identity_block, 32, 32,3,1)

        self.block0_3 = self._make_residual_block_(conv_block, 32, 64,3,1)
        self.block1_3 = self._make_residual_block_(identity_block, 64, 64,3,1)

        self.block0_4 = self._make_residual_block_(conv_block, 64, 64,3,1)
        self.block1_4 = self._make_residual_block_(identity_block, 64, 64,3,1)

        self.lin_layer = nn.Sequential(nn.Linear(10816, 4096),nn.Sigmoid())

    def _make_residual_block_(self, block, c_in, c_out,kernel_size,padding):
        layers = []
        layers.append(block(c_in,c_out,kernel_size,padding))

        return nn.Sequential(*layers)  



    def forward_once_print(self, x):
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
#layer1
        out = self.block0_3(out) 
        print("block0_3",out.shape)
        out = self.block1_3(out) 
        print("block1_3",out.shape)
#layer2
        out = self.block0_4(out) 
        print("block0_4",out.shape)
        out = self.block1_4(out)
        print("block1_4",out.shape)

        out = out.view(out.size()[0], -1)  ##=> euqals flatten
        print("view",out.shape)
        out = self.lin_layer(out)
        print("lin_layer=output",out.shape)

        return out

    def forward_once(self, x):
        out = self.prep(x)
        #layer1
        out = self.block0_1(out) 
        out = self.block1_1(out)
        #out = self.block1_1(out) 
        #layer2
        out = self.block0_2(out) 
        out = self.block1_2(out)
        #out = self.block1_2(out)
        #out = self.block1_2(out)
        #layer3
        out = self.block0_3(out) 
        out = self.block1_3(out)
        #out = self.block1_3(out)
        #out = self.block1_3(out)
        #layer4
        out = self.block0_4(out) 
        out = self.block1_4(out)
        #out = self.block1_4(out)

        out = out.view(out.size()[0], -1)
        out = self.lin_layer(out)

        return out

    def forward(self, x,y):
        out_x = self.forward_once(x)
        out_y = self.forward_once(y)
        #out  = torch.abs((out_x-out_y))
        #out  = self.final(out)

        return out_x, out_y

