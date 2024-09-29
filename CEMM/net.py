# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlad import NetVLAD_group as VLAD

# 3D Convolution Block
class d_conv(nn.Module):
    def __init__(self, ch1, ch2, kernel_size, stride=1):
        super(d_conv, self).__init__()
        self.conv1 = nn.Conv3d(ch1, ch1, kernel_size=(1, kernel_size, kernel_size),
                               padding=(0, kernel_size // 2, kernel_size // 2),
                               stride=(1, stride, stride), groups=ch1)
        self.conv2 = nn.Conv3d(ch1, ch2, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# Block with Batch Normalization
class b_conv3d(nn.Module):
    def __init__(self, ch1, ch2, kernel_size=3):
        super(b_conv3d, self).__init__()
        self.dconv1 = d_conv(ch1, ch2, kernel_size=kernel_size)
        self.bn = nn.BatchNorm3d(ch2)

    def forward(self, x):
        x = self.dconv1(x)
        return F.relu(x)  # BatchNorm is commented out


# Main Network Architecture
class boneNet(nn.Module):
    def __init__(self, ch_list=None):
        super(boneNet, self).__init__()
        ch_list = ch_list or [1, 16, 32, 64, 96]  # Default channel sizes
        self.block1 = nn.Sequential(
            nn.Conv3d(ch_list[0], ch_list[1], kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(ch_list[1], ch_list[1], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(ch_list[1])
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block2 = b_conv3d(ch_list[1], ch_list[2])
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block3 = b_conv3d(ch_list[2], ch_list[3])
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block4 = b_conv3d(ch_list[3], ch_list[4])
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.v_len = ch_list[-1]  # Last channel size

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        return self.global_pool(x)  # Global pooling


class Net(nn.Module):
    def __init__(self, frame_num, cluster_size=20, output_dim=64):
        super(Net, self).__init__()
        self.bone = boneNet()
        self.frame_num = frame_num
        self.vlad = VLAD(feature_size=self.bone.v_len, max_frames=self.frame_num,
                         cluster_size=cluster_size, groups=2, expansion=2)
        self.linear1 = nn.Linear(cluster_size * self.bone.v_len, output_dim)
        self.classifier_sig = nn.Sequential(
            nn.Linear(output_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        ap_img, pp_img, vp_img = x
        ap_out = self.bone(ap_img).view(ap_img.size(0), -1)
        pp_out = self.bone(pp_img).view(pp_img.size(0), -1)
        vp_out = self.bone(vp_img).view(vp_img.size(0), -1)

        out = torch.cat([ap_out, pp_out, vp_out], dim=1)
        out = self.vlad(out)
        out = self.linear1(out)
        return self.classifier_sig(out)  # Sigmoid classifier