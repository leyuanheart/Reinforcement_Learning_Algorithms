#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:20:23 2020

@author: leyuan
"""

import math


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# torch.cuda.is_available()

class MLP_pytorch(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(MLP_pytorch, self).__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.action_dim)
        
    def forward(self, obs):   # obs is set to be tensor
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    


class DuelingNetwork_pytorch(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(DuelingNetwork_pytorch, self).__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # value head
        self.value = nn.Linear(64, 1)
        # advantage head
        self.adv = nn.Linear(64, self.action_dim)
        
        # 也有把这连个写到一起的
        # self.out = nn.Linear(64, self.action_dim + 1)
        # 但是我总觉得这样不算是真正地解耦value和advantage函数
        
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value(x)
        adv = self.adv(x)
        
        adv = adv - adv.mean()     # forward() 是按照单个输入的逻辑来写的，即batch size=1，当然也可以写成 adv.mean(dim=-1, keepdim=True)
        
        return value + adv

    
    
class CONV_pytorch(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(CONV_pytorch, self).__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.action_dim)
        
    def forward(self, obs):     # x should be torch.float type
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def feature_size(self):        
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).shape[1]




class NoisyLinear(nn.Module):
    '''
    如果是使用Noise on parameter这个trick的话，那整个构建Network的layer就不再是Pytorch自带的nn.Linear(),
    而是使用这个自定义的NoisyLinear()
    y = w*x + b
    重参数化：
    将参数w, b看成是均值为\mu，方差为\sigma的正态分布，同时带了随机噪声，
    \mu，\sigma是网络用来训练的参数
    '''
    def __init__(self, in_features, out_features, std_init=0.4, factorised_noise=True):
        super(NoisyLinear ,self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorised_noise = factorised_noise
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        '''
        torch.Module.register_buffer是另一种构造参数的方法，这种方法构造的参数不参与梯度传播，
        但是会在model.state_dict()
        '''
        self.bias_mu = nn.Parameter(torch.empty(out_features))        
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        '''
        初始化参数
        '''
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))


    def _scale_noise(self, size):
        '''
        将每个随机数的方差都变成 1
        mul_(), sqrt_()的作用等同于 Tensor.mul(), Tensor.sqrt()
        '''
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


    def sample_noise(self):
        if self.factorised_noise:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        else:
            self.weight_epsilon.copy_(torch.randn((self.out_features, self.in_features)))
            self.bias_epsilon.copy_(torch.randn(self.out_features))
    
    
    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)




class CategoricalNetwork_pytorch(nn.Module):
    '''
    并不是只写一个Categorical的Network就可以了，
    涉及到projection和计算loss的部分还得参考一下论文才行
    '''
    def __init__(self, input_shape, action_dim, atoms=51):
        super(CategoricalNetwork_pytorch, self).__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.atoms = atoms
        
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.action_dim * self.atoms)
        
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        logits = self.fc4(x)
        logits = logits.view(-1, self.action_dim, self.atoms)
        
        probs = F.softmax(logits, dim=2)
        
        return probs
        



class QuantileNetwork_pytorch(nn.Module):
    '''
    并不是只写一个Quantile的Network就可以了，
    计算loss的部分还得参考一下论文才行
    '''
    def __init__(self, input_shape, action_dim, quantiles=51):
        super(QuantileNetwork_pytorch, self).__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.quantiles = quantiles
        
        
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.action_dim*self.quantiles)
        
    def forward(self, obs):   # obs is set to be tensor
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x.view(-1, self.action_dim, self.quantiles)
        



class QuantileDuelingNetwork_pytorch(nn.Module):
    def __init__(self, input_shape, action_dim, quantiles=51):
        super(QuantileNetwork_pytorch, self).__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.quantiles = quantiles
        
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # value head
        self.value = nn.Linear(64, self.quantiles)
        # advantage head
        self.adv = nn.Linear(64, self.action_dim*self.quantiles)
        
        
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value(x)
        adv = self.adv(x)
        adv = adv.view(-1, self.action_dim, self.quantiles)
        
        final = value.view(-1, 1, self.quantiles) + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)   
        # final shape: (-1, self.action_dim, self.quantiles)
        
        # q value
        # q_vals = final.mean(dim=-1)
        
        return final






















