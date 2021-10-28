class Discriminator(nn.Module):

    def __init__(self, conv_dim):

        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        
        self.leaky_relu_slope = 0.2

        #First layer with input size equal to the image size
        self.conv1 = conv(3, conv_dim, norm=False) 
        #Second layer in which the size is reduced to 8x8 
        self.conv2 = conv(conv_dim, conv_dim*2)       
        #Third layer in which the size is reduced to 4x4 
        self.conv3 = conv(conv_dim*2, conv_dim*4)         
                
        #Final classification fully connected layer
        self.fc = nn.Linear(conv_dim*4*4*4, 1)
    
    
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), self.leaky_relu_slope)
        out = F.leaky_relu(self.conv2(out), self.leaky_relu_slope)
        out = F.leaky_relu(self.conv3(out), self.leaky_relu_slope)
        
        #flattening: output is converted into the single stream of data
        out = out.view(-1, self.conv_dim*4*4*4) 

        out = self.fc(out) 
        
        return out