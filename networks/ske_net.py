import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
import math
class ChannelAttention_H(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention_H, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)  # C1D 一维卷积
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # 根据通道数求一维卷积大卷积核大小 odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out
class SpatialAttention_H(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention_H, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate 论文中经过实验发现0.6效果最佳
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # 根据通道数以及分离率确定重要通道的数量 even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    
    def get_im_subim_channels(self, C_im, M): # 根据Channel_Attention_Map得到重要通道以及不重要的通道
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature
    
class ResBlock_HAM(nn.Module):
    def __init__(self, Channel_nums):
        super(ResBlock_HAM, self).__init__()
        self.ChannelAttention = ChannelAttention_H(Channel_nums)
        self.SpatialAttention = SpatialAttention_H(Channel_nums)

    def forward(self, x_in):
        channel_attention_map = self.ChannelAttention(x_in)
        channel_refined_feature = channel_attention_map * x_in
        final_refined_feature = self.SpatialAttention(channel_refined_feature, channel_attention_map)
        out = final_refined_feature
        return out
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = DoubleConv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = DoubleConv2d(512, 1024, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.att1 = SCSEModule(64)
        self.att2 = SCSEModule(128)
        self.att3 = SCSEModule(256)
        self.att4 = SCSEModule(512)
        self.att5 = SCSEModule(1024)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.att1(out1)

        out2 = self.conv2(self.pooling(out1))
        out2 = self.att2(out2)

        out3 = self.conv3(self.pooling(out2))
        out3 = self.att3(out3)

        out4 = self.conv4(self.pooling(out3))
        out4 = self.att4(out4)

        out5 = self.conv5(self.pooling(out4))
        out5 = self.att5(out5)

        return out1, out2, out3, out4, out5


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = UpConv2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = UpConv2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = UpConv2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = UpConv2d(128, 64, kernel_size=2, stride=2)

        self.conv1 = DoubleConv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(128, 64, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_128 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_64 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_32 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.cbam1 = ResBlock_HAM(512)
        self.cbam2 = ResBlock_HAM(256)
        self.cbam3 = ResBlock_HAM(128)
        self.cbam4 = ResBlock_HAM(64)


    def forward(self, out1, out2, out3, out4, x):
        x = self.upconv1(x)
        x = torch.cat([x, out4], dim=1)
        x = self.conv1(x)
        x = self.cbam1(x)
        aux_32 = self.aux_conv_32(x)

        x = self.upconv2(x)
        x = torch.cat([x, out3], dim=1)
        x = self.conv2(x)
        x = self.cbam2(x)
        aux_64 = self.aux_conv_64(x)

        x = self.upconv3(x)
        x = torch.cat([x, out2], dim=1)
        x = self.conv3(x)
        x = self.cbam3(x)
        aux_128 = self.aux_conv_128(x)

        x = self.upconv4(x)
        x = torch.cat([x, out1], dim=1)
        x = self.conv4(x)
        x = self.cbam4(x)
        x = self.conv1x1(x)

        return x, aux_128, aux_64, aux_32


class UnetAttention(nn.Module):
    def __init__(self):
        super(UnetAttention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        if (self.training):
            out1, out2, out3, out4, x = self.encoder(x.float())
            x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

            return x, aux_128, aux_64, aux_32
        else:
            out1, out2, out3, out4, x = self.encoder(x.float())
            x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)
            return x
    
# https://github.com/namdvt/skeletonization/blob/master/model/unet_att.py