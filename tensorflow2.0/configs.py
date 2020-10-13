#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:54:05 2020

@author: leyuan
"""


import numpy as np


class Config(object):
    def __init__(self):
        
        
        #epsilon variables    espislon-greedy exploration
        self.epsilon_start = 0.1
        self.epsilon_final = 0.1
        self.epsilon_decay = 30000
        # epsilon随着traning step增加递减, 探索的越来越少，
        # 但是这不是对于SARSA来说才需要的吗，对于Q-learning，behavior policy可以保持一定的探索程度
        # 留个疑问，之后看看其他大佬的代码是怎么处理这一部分的
        self.epsilon_by_step = lambda step: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)
        
        # misc agent variables
        self.gamma = 0.99
        self.training_env_seed = 123
        
        # memory
        self.target_net_update_freq = 1000 # 每个多少的epoch更新target网络
        self.exp_replay_size = 100000      # replay memory的容量
        self.batch_size = 32
        # prioritized replay memory
        self.alpha = 0.6 
        self.beta_start = 0.4 
        self.beta_steps = 100000
        
        
        # learning control variables
        self.learn_start = 1000     # 在replay memory中预先放入多少个tuple再开始训练
        self.lr = 1e-4
        self.max_steps = 1000000    # 训练多少个steps
        