# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD_group(nn.Module):
    def __init__(self, feature_size, max_frames, cluster_size, groups,expansion):
        super().__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.groups = groups
        self.expansion = expansion
        
        self.new_feature_size = self.expansion * self.feature_size // self.groups
        
        self.fc1 = nn.Linear(self.feature_size, self.expansion * self.feature_size)
        nn.init.kaiming_uniform_(self.fc1.weight, a=2, mode='fan_in')
        self.fc2 = nn.Linear(self.expansion * self.feature_size, self.groups)
        nn.init.kaiming_uniform_(self.fc2.weight, a=2, mode='fan_in')

        self.cluster_weights = nn.Parameter(torch.rand(self.expansion * self.feature_size, self.groups * self.cluster_size))
        nn.init.kaiming_uniform_(self.cluster_weights, a=2, mode='fan_in')
        self.bn_activation = nn.BatchNorm1d(self.groups * self.cluster_size, affine=True)
        self.cluster_weight2 = nn.Parameter(torch.rand(1, self.new_feature_size, self.cluster_size))
        nn.init.kaiming_uniform_(self.cluster_weight2, a=2, mode='fan_in')
        self.bn_vlad=nn.BatchNorm1d(self.new_feature_size * self.cluster_size, affine=True)
        
        # b*d,c
    def forward(self,x):
        x = x.view(-1, self.feature_size)
        # b*d,c  -->  b*d,expansion*c
        x = self.fc1(x)
        
        # b*d,c  --> b*d,groups
        attention = self.fc2(x)
        # b*d,groups  -->  b,d*groups,1
        attention = attention.view(-1, self.max_frames, self.groups)
        attention = attention.view(-1, self.max_frames * self.groups).unsqueeze(-1)

        # b*d,expansion*c  and  expansion*c,groups*cluster_size  -->  b*d,groups*cluster_size
        activation = torch.matmul(x, self.cluster_weights)
        activation = self.bn_activation(activation)
        # b*d,groups*cluster_size  -->  b,d*groups,cluster_size
        activation = activation.view(-1, self.max_frames, self.groups, self.cluster_size)
        activation = activation.view(-1, self.max_frames * self.groups, self.cluster_size)
        activation = F.softmax(activation, dim=-1)
        # b,d*groups,cluster_size  and  b,d*groups,1  --> b,d*groups,cluster_size
        activation=torch.mul(activation,attention)

        # b,d*groups,cluster_size  --> b,1,cluster_size
        a_sum = activation.sum(dim=-2,keepdim=True)
        # b,1,cluster_size  and  1,c',cluster_size  -->  b,c',cluster_size
        a = a_sum.mul(self.cluster_weight2)

        # b,d*groups,cluster_size  -->  b,cluster_size,d*groups
        activation = activation.transpose(1,2).contiguous()
        # b*d,expansion*c  -->  b,d*groups,c'
        x = x.reshape(-1, self.max_frames * self.groups, self.new_feature_size)
        
        # b,cluster_size,d*groups  and  b,d*groups,c'  -->  b,cluster_size,c'
        vlad = torch.matmul(activation,x)
        # b,cluster_size,c'  -->  b,c',cluster_size
        vlad = vlad.transpose(1,2).contiguous()
        # b,c',cluster_size  and  b,c',cluster_size  --> b,c',cluster_size
        vlad = vlad.sub(a)
        
        vlad = F.normalize(vlad,p=2,dim=1)
        vlad = vlad.view(-1,self.new_feature_size*self.cluster_size)
        vlad = self.bn_vlad(vlad)
        
        return vlad