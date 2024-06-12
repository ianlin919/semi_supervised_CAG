import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv = ConvBlock((in_channels1 + in_channels2), out_channels, dropout_p)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = nn.functional.interpolate(
                x1,scale_factor=2, mode='nearest-exact',
            )
        else:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output, x

class Proj_Module(nn.Module):
    def __init__(self, params, kernel_size=4, stride=4, padding=0):
        super(Proj_Module, self).__init__()
        self.ft_chns = params['feature_chns']
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.proj_1 = nn.Sequential(
            nn.Conv2d(self.ft_chns[0], self.ft_chns[2], 
                      kernel_size=self.kernel_size, 
                      stride=self.stride, 
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.ft_chns[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ft_chns[2], self.ft_chns[2], 
                      kernel_size=self.kernel_size, 
                      stride=self.stride, 
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.ft_chns[2]),
            nn.ReLU(inplace=True)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2d(self.ft_chns[2], self.ft_chns[3], 
                      kernel_size=self.kernel_size, 
                      stride=self.stride, 
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.ft_chns[3]),
            nn.ReLU(inplace=True)
        )
    def forward(self, feature):
        out = self.proj_2(self.proj_1(feature))
        return out
    
class Proj_UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(Proj_UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.proj_final = Proj_Module(params)

    def forward(self, x, get_patch=False):
        if get_patch:
            p_x, unfold_shape = self.split_patch(x)
            p_output, p_final = self.decoder(self.encoder(p_x))
            output = self.merge_patch(p_output, unfold_shape)
            final = self.merge_patch(p_final, unfold_shape)
            print(p_output.shape, p_final.shape)
        else:
            feature = self.encoder(x)
            output, final = self.decoder(feature)
            print(output.shape, final.shape)
            
        if (self.training):
            output_proj = self.proj_final(final)
            return output, output_proj
        return output
    
    def split_patch(self, orig):
        assert orig.size(-2) == orig.size(-1)
        cs, cn = 64, 64
        
        patches = orig.unfold(2, cs, cn).unfold(3, cs, cn)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, 1,cs, cn)
        
        return patches, unfold_shape
        
    def merge_patch(self, patches, unfold_shape):
        _, C, _, _ = patches.size()
        B_p, _, H_p, W_p, H, W = unfold_shape
        patches_orig = patches.view((B_p, C, H_p, W_p, H, W))
        output_h = unfold_shape[2] * unfold_shape[4]
        output_w = unfold_shape[3] * unfold_shape[5]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, output_h , output_w)
        
        return patches_orig