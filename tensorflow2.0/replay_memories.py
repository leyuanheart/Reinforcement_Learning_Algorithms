#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:19:27 2020

@author: leyuan
"""

import numpy as np
import random


class ExperienceReplayMemory(object):
    '''
    最简单的replay memory，只包括简单的添加（add）和抽取（sample）功能
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def add(self, transition):
        # transition = (obs_t, action, reward, obs_tp1, done)
        '''
        这里考虑还是以一个transition变量作为参数，
        这样普适性比直接用obs_t, action, reward, obs_tp1, done更高，
        可以自己定义transition所包含的内容
        '''
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        sub_buffer = random.sample(self.memory, batch_size)
        return sub_buffer
        # obses_t, actions, rewards, obses_tp1, dones = zip(*sub_buffer)
        # return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
   
    def __len__(self):
        return len(self.memory)




class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=100000):
        '''
        alpha 是从replay memory里采样的权重中的超参数，alpha=0表示等权重抽样
        beta 是用来纠正期望偏差的Importance weight里的超参数， beta=1表示完全纠偏
        这一种是用 |TD error|+epsilon来构造抽样的权重，
        还有一种是用|TD error|的rank来构造权重，代码更加复杂，需要引入一种 SumTree 的数据结构
        '''
        self.prob_alpha = alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.step = 1
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        
    def beta_by_step(self, step):
        '''
        beta逐渐增大到 1
        '''
        return min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_steps)
        

    def add(self, transition):
        max_prior = self.priorities.max() if self.buffer else 1.0**self.prob_alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = max_prior
        
        self.pos = (self.pos + 1) % self.capacity
        
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priors = self.priorities
        else:
            priors = self.priorities[:self.pos]
            
        total = len(self.buffer)
        
        probs = priors / priors.sum()
        
        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_step(self.step)
        self.step += 1
        
        # compute the importance weight
        # minimize of all probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total)**(-beta)
        
        weights = (total * probs[indices])**(-beta)
        weights /= max_weight
        
        return samples, indices, weights
        
    
    def update_priorities(self, batch_indices, batch_tderror_abs):
        for idx, tderror_abs in zip(batch_indices, batch_tderror_abs):
            self.priorities[idx] = (tderror_abs + 1e-5)**self.prob_alpha
        
   
    def __len__(self):
        return len(self.buffer)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        