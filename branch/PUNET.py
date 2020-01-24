import torch
import torch.nn as nn

OUT_CHANNELS = 1        # output channel
DROP_OUT_RATE = 0.6     # drop_out_rate

# assistant block
class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv => Batch_Norm => ReLU
    '''
    def __init__(self, in_channel, out_channel):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels  = in_channel,     # input height
                out_channels = out_channel,    # n_filters
                kernel_size  = 3,              # filter size
                stride       = 1,              # filter movement/step
                padding      = 1,              # the same width and length of this image after Conv
                bias         = False,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=DROP_OUT_RATE),
            nn.Conv3d(
                in_channels  = out_channel,
                out_channels = out_channel,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                bias         = False,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    ''' normal down path
        MaxPool3d => double_conv
    '''
    def __init__(self, in_channel, out_channel):
        super(inconv, self).__init__()
        self.inputconv = double_conv(in_channel, out_channel)

    def forward(self, x):
        x = self.inputconv(x)
        return x

class downp(nn.Module):
    ''' normal down path
        MaxPool3d => double_conv
    '''
    def __init__(self, in_channel, out_channel):
        super(downp, self).__init__()
        self.down = nn.MaxPool3d(2)
        self.conv = double_conv(in_channel+64, out_channel)

    def forward(self, x, ini):
        x = self.down(x)
        x = torch.cat([x,ini], dim=1)
        x = self.conv(x)
        return x

class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_channel, out_channel):
        super(up, self).__init__()

        self.up = nn.ConvTranspose3d(in_channel, in_channel//2, kernel_size=4, stride=2, padding=1)
        self.conv = double_conv(in_channel, out_channel)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

class decoder(nn.Module):
    ''' dencoder path
        conv_transpose => double_conv
    '''
    def __init__(self, in_channel, out_channel):
        super(decoder, self).__init__()

        self.decoder = nn.ConvTranspose3d(in_channel, in_channel, kernel_size=4, stride=2, padding=1)
        self.conv = double_conv(in_channel, out_channel)

    def forward(self, x):

        x = self.decoder(x)
        x = self.conv(x)
        return x

# model architecture
class Unet_3d(nn.Module):
    def __init__(self, in_channel):
        super(Unet_3d, self).__init__()
        self.in_channels  = in_channel
        self.out_channels = OUT_CHANNELS

        # input layer
        self.inputlayer0  = inconv(self.in_channels, 64)
        self.inputlayer1  = inconv(self.in_channels, 64)
        self.inputlayer2  = inconv(self.in_channels, 64)
        self.inputlayer3  = inconv(self.in_channels, 64)

        # down-sampling path
        self.downlayer1  = downp(64,  128)
        self.downlayer2  = downp(128, 256)
        self.downlayer3  = downp(256, 512)

        # up-sampling  path
        self.uplayer3 = up(512, 256)
        self.uplayer2 = up(256, 128)
        self.uplayer1 = up(128, 64)
        self.outputlayer = nn.Conv3d(64, self.out_channels,3,1,1)

        # auxiliary path
        self.avgpool =  nn.AvgPool3d(2)
        self.auxlayer2 = nn.Sequential(
            decoder(256, self.out_channels),
            decoder(self.out_channels, self.out_channels),
        )
        self.auxlayer1 = decoder(128, self.out_channels)

    
    def forward(self, x, ht=None):
        #image pyramid
        in1 = self.avgpool(x)
        in2 = self.avgpool(in1)
        in3 = self.avgpool(in2)

        #feature pyramid
        in1 = self.inputlayer1(in1)
        in2 = self.inputlayer2(in2)
        in3 = self.inputlayer3(in3)

        # down-sampling path
        x0 = self.inputlayer0(x)
        x1 = self.downlayer1(x0,in1)
        x2 = self.downlayer2(x1,in2)
        x3 = self.downlayer3(x2,in3)

        # up-sampling  path
        y2 = self.uplayer3(x3,x2)
        y1 = self.uplayer2(y2,x1)
        y0 = self.uplayer1(y1,x0)

        # auxiliary prediction
        aux2 = self.auxlayer2(y2)
        aux1 = self.auxlayer1(y1)
        y    = self.outputlayer(y0)

        pred_prob = torch.sigmoid(y)
        aux2_prob = torch.sigmoid(aux2)
        aux1_prob = torch.sigmoid(aux1)
        
        return pred_prob, aux2_prob, aux1_prob


