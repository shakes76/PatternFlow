#Convolution layer function for the discriminator
def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):

    conv_layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    conv_layers.append(conv_layer)
    if norm:
        conv_layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*conv_layers)