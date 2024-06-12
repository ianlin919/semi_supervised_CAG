# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

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
        return output

def _l2_normalize(d):
        """Normalizing per batch axis"""
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

def _get_r_adv(x_list, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training according to
    https://arxiv.org/abs/1704.03976
    """
    decoder.eval()
    x_detached = [item.detach() for item in x_list]
    xe_detached = x_detached[-1]
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x_list[-1].shape).sub(0.5).to(x_list[-1].device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        x_detached[-1] = xe_detached + xi * d
        pred_hat = decoder(x_detached)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    decoder.train()
    return x_list[-1] + r_adv

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x
    
def get_aux(feature, decoder, aux_name):
    if aux_name == 'FeatureNoise':
        aux_feature = [FeatureNoise()(i) for i in feature]
    elif aux_name == 'FeatureDropout':
        aux_feature = [FeatureDropout(i) for i in feature]
    elif aux_name == 'Dropout':
        aux_feature = [Dropout(i) for i in feature]
    elif aux_name == 'VAT':
        aux_feature = feature
        aux_feature_ = _get_r_adv(feature, decoder, it=1, xi=0.1, eps=10)
        aux_feature[-1] = aux_feature_
    else:
        aux_feature = feature
    return aux_feature

class PCPS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(PCPS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder1 = Encoder(params)
        self.encoder2 = Encoder(params)
        self.decoder1 = Decoder(params)
        self.decoder2 = Decoder(params)
        self.aux_names = list(set(['FeatureNoise', 'FeatureDropout', 'Dropout', 'VAT', 'None']))
        
    def forward(self, x):
        if (self.training):
            feature1 = self.encoder1(x)
            feature2 = self.encoder2(x)
            
            aux_name_1 = random.sample(self.aux_names, k=2)
            # aux_name_2 = random.choices(self.aux_names, k=1)
            
            aux_feature1 = get_aux(feature1, self.decoder1, aux_name_1[0])
            aux_feature2 = get_aux(feature2, self.decoder2, aux_name_1[1])
            
            output1, aux_output1 = self.decoder1(feature1), self.decoder1(aux_feature1)
            output2, aux_output2 = self.decoder2(feature2), self.decoder2(aux_feature2)
            
            return output1, output2, aux_output1, aux_output2
        else:
            feature = self.encoder1(x)
            output = self.decoder1(feature)
            return output
        
class PCPS2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(PCPS2, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.aux_names = list(set(['FeatureNoise', 'FeatureDropout', 'Dropout', 'VAT', 'None']))
        
    def forward(self, x, aux_name=None):
        feature = self.encoder(x)
        if (self.training):
            if (aux_name is not None) and (aux_name in self.aux_names):
                aux_feature = get_aux(feature, self.decoder, aux_name)
                output, aux_output = self.decoder(feature), self.decoder(aux_feature)
                return output, aux_output
            else:
                output = self.decoder(feature)
                return output
        else:
            output = self.decoder(feature)
            return output