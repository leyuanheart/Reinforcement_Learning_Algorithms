#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:01:42 2020

@author: didi
"""


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta
from IPython.display import clear_output


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import gym

from configs import Config
from replay_memories import ExperienceReplayMemory, PrioritizedReplayMemory
from networks_pytorch import MLP_pytorch, DuelingNetwork_pytorch, CONV_pytorch, CategoricalNetwork_pytorch, QuantileNetwork_pytorch

config = Config()

class DQNAgentPytorch(object):
        def __init__(self, env_name=None, network=MLP_pytorch, double=False, prioritized=False, n_steps=1, eval_mode=False, config=config):
            
            self.env_name = env_name
            self.env = gym.make(env_name)
            self.obs_dim = self.env.observation_space.shape[0]   # 根据环境来设置
            self.action_dim = self.env.action_space.n
            self.eval_mode = eval_mode
            
            self.gamma = config.gamma
            self.lr = config.lr
            self.target_net_update_freq = config.target_net_update_freq
            self.experience_replay_size = config.exp_replay_size
            self.batch_size = config.batch_size
            self.learn_start = config.learn_start     
            
            self.prioritized = prioritized
            self.alpha = config.alpha
            self.beta_start = config.beta_start
            self.beta_steps = config.beta_steps
            self.n_steps = n_steps
            self.n_step_buffer = []
            
            if self.prioritized:
                self.memory = PrioritizedReplayMemory(self.experience_replay_size, self.alpha, self.beta_start, self.beta_steps)
            else:
                self.memory = ExperienceReplayMemory(self.experience_replay_size)
            
            
            self.network = network
            self.double = double
            
    
            self.model = self.network(self.obs_dim, self.action_dim)
            self.target_model = self.network(self.obs_dim, self.action_dim)
            self.target_model.load_state_dict(self.model.state_dict())
            
            # train和eval模式的差别主要是Batch Normalization和Dropout的使用差别
            if self.eval_mode:
                self.model.eval()
                self.target_model.eval()
            else:
                self.model.train()
                self.target_model.train()
            
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
            self.loss = nn.SmoothL1Loss(reduction='none')
                    
            self.update_count = 0
            self.losses = []
            self.rewards = []
            self.episode_length = []
            
        
        def train(self, step=0):
            if self.eval_mode:
                return None
            
            if step < self.learn_start:
                return None
            
            if self.prioritized:
                transitions, indices, weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
            
            # if self.prioritized:
            #     obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            # else:
            #     obses_t, actions, rewards, obses_tp1, dones = transitions
            obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            
            obses_t = torch.tensor(obses_t, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long).squeeze().view(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float).squeeze().view(-1, 1)
            obses_tp1 = torch.tensor(obses_tp1, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.int32).squeeze().view(-1, 1)  
            
            
            # compute loss
            chosen_q_vals = self.model(obses_t).gather(axis=1, index=actions)
            
            with torch.no_grad():
                if self.double:
                    actions_tp1 = self.model(obses_tp1).max(dim=1)[1].view(-1, 1)
                    q_tp1_vals = self.target_model(obses_tp1).gather(axis=1, index=actions_tp1)
                else:
                    q_tp1_vals = self.target_model(obses_tp1).max(dim=1)[0].view(-1, 1)
                
                '''
                这里要注意V(s')的折现步数
                '''
                targets = rewards + self.gamma**self.n_steps * q_tp1_vals * (1 - dones)
            
            if self.prioritized:
                loss = self.loss(chosen_q_vals, targets) * torch.tensor(weights)
                diff = chosen_q_vals - targets
                self.memory.update_priorities(indices, diff.detach().squeeze().abs().numpy().tolist())
                
            else:
                loss = self.loss(chosen_q_vals, targets)
            
            loss = loss.mean()
            
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():      # 梯度裁剪非常有用！！！
                param.grad.data.clamp(-1, 1)
            self.optimizer.step()
                   
            self.losses.append(loss.item())
            
            # update target model
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            
            # 这一段可以放在training loop里
            # if self.update_count % 10000 == 0:
            #     mean_returns = self.eval(5)
            #     if mean_returns > 300:
            #         self.render()
                
        
        def eval_(self, n_trajs):
            env = gym.make(self.env_name)
            self.eval = True
            episode_return = 0
            episode_length = 0            
            for _ in range(n_trajs):
                obs = env.reset()
                for _ in range(1000):
                    a = self.get_action(obs)
                    obs, reward, done, info = env.step(a)
                    episode_return += reward
                    episode_length += 1
                    
                    if done:
                        self.rewards.append(episode_return)
                        self.episode_length.append(episode_length)
                        episode_return = 0
                        episode_length = 0
                        break
                        
            # print('eval {} trajs, mean return: {}'.format(n_trajs, np.mean(episode_returns)))
            env.close()
            self.eval = False
            return np.mean(self.rewards[-n_trajs:]), np.max(self.rewards[-n_trajs:]), np.mean(self.episode_length[-n_trajs:]), np.max(self.episode_length[-n_trajs:])
        

                
        def get_action(self, obs, eps=0.1):   # epsilon-greedy policy
            with torch.no_grad():
                if np.random.random() >= eps or self.eval_mode:
                    # print(s.dtype)
                    obs = np.expand_dims(obs, 0)
                    obs = torch.tensor(obs, dtype=torch.float)
                    a = self.model(obs).max(dim=1)[1]
                    return a.item()
                else:
                    return np.random.randint(0, self.action_dim)
        
    
        def n_steps_replay(self, transition):
            '''
            如果想要使用 n-steps TD learning,  在收集transition时就用这个函数，而不是用self.memory.add()
            模拟证明还是挺有用的，在LunarLander-v2上，其他DQN tricks都不加，只是用n-steps=5,就能训练得蛮好的，
            虽然前期的loss趋势很诡异，会先上升一段，然后开始慢慢下降 (可能是由于当时计算target的时候V(s')的折现只考虑了一步，已经改正）
            '''
            _, _, _, obs_tpn, done = transition
            self.n_step_buffer.append(transition)
            
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            R = sum([self.n_step_buffer[i][2] * self.gamma**i for i in range(self.n_steps)])
            obs_t, action, _, _, _ = self.n_step_buffer.pop(0)
            
            self.memory.add((obs_t, action, R, obs_tpn, done))
        
        
        
        def save_w(self):
            # Returns a dictionary containing a whole state of the module.
            torch.save(self.model.state_dict(), './model.dump')
            torch.save(self.optimizer.state_dict(), './optim.dump')
            
        def load_w(self):
            fname_model = './model.dump'
            fname_optim = './optim.dump'
            
            if os.path.isfile(fname_model):
                self.model.load_state_dict(torch.load(fname_model))
                self.target_model.load_state_dict(self.model.state_dict())
                
            if os.path.isfile(fname_optim):
                self.optimizer.load_state_dict(torch.load(fname_optim))
                
        def save_replay(self):
            pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
            
        def load_replay(self):
            fname = './exp_replay_agent.dump'
            if os.path.isfile(fname):
                self.memory = pickle.load(open(fname, 'rb'))
                
                
        def huber_loss(self, x, delta=1):    # x must be tensor
            cond = (x.abs() <= delta).to(torch.float32)
            return 0.5 * x.pow(2) * cond + delta * (x.abs() - 0.5 * delta) * (1 - cond)
        
        
        def render(self):
            env = gym.make(self.env_name)
            self.eval = True
            obs = env.reset()
            for _ in range(1000):
                env.render()
                a = self.get_action(obs)
                obs, reward, done, info = env.step(a)
                if done:
                    break
            env.close()
            self.eval = False
        
        



class CategoricalDQNAgentPytorch(object):   
    
        def __init__(self, env_name=None, network=CategoricalNetwork_pytorch, 
                     atoms = 51, v_min = -10, v_max = 10,
                     prioritized=False, n_steps=1, eval_mode=False, config=config):
            
            self.env_name = env_name
            self.env = gym.make(env_name)
            self.obs_dim = self.env.observation_space.shape[0]   # 根据环境来设置
            self.action_dim = self.env.action_space.n
            self.eval_mode = eval_mode
            
            
            self.atoms = atoms
            self.v_min = v_min
            self.v_max = v_max
            self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms)
            self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
            
            self.gamma = config.gamma
            self.lr = config.lr
            self.target_net_update_freq = config.target_net_update_freq
            self.experience_replay_size = config.exp_replay_size
            self.batch_size = config.batch_size
            self.learn_start = config.learn_start     
            
            self.prioritized = prioritized
            self.alpha = config.alpha
            self.beta_start = config.beta_start
            self.beta_steps = config.beta_steps
            self.n_steps = n_steps
            self.n_step_buffer = []
            
            if self.prioritized:
                self.memory = PrioritizedReplayMemory(self.experience_replay_size, self.alpha, self.beta_start, self.beta_steps)
            else:
                self.memory = ExperienceReplayMemory(self.experience_replay_size)
            
            
            self.network = network
            # self.double = double
            
    
            self.model = self.network(self.obs_dim, self.action_dim, self.atoms)
            self.target_model = self.network(self.obs_dim, self.action_dim, self.atoms)
            self.target_model.load_state_dict(self.model.state_dict())
            
            # train和eval模式的差别主要是Batch Normalization和Dropout的使用差别
            if self.eval_mode:
                self.model.eval()
                self.target_model.eval()
            else:
                self.model.train()
                self.target_model.train()
            
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
            # loss 自己在下面写
                    
            self.update_count = 0
            self.losses = []
            self.rewards = []
            self.episode_length = []
            
            
        
        def train(self, step=0):
            if self.eval_mode:
                return None
            
            if step < self.learn_start:
                return None
            
            if self.prioritized:
                transitions, indices, weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
            
            
            # if self.prioritized:
            #     obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            # else:
            #     obses_t, actions, rewards, obses_tp1, dones = transitions
            obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            
            obses_t = torch.tensor(obses_t, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long).squeeze().view(-1, 1, 1).expand(-1, -1, self.atoms)
            rewards = torch.tensor(rewards, dtype=torch.float).squeeze().view(-1, 1, 1)
            obses_tp1 = torch.tensor(obses_tp1, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.int32).squeeze().view(-1, 1, 1)  
                        
            
            #============= compute loss ===================              
            with torch.no_grad():  
                # compute Q value
                next_distribution = self.model(obses_tp1) * self.supports
                q_vals = next_distribution.sum(dim=-1)
                
                
                # chose the action
                actions_tp1 = q_vals.max(dim=1)[1].view(-1, 1, 1).expand(-1, -1, self.atoms)  # 默认使用 double DQN 的思想，用model选动作，用target model算Q value
                '''
                这里reshape和expand维度有些绕，需要捋一捋，
                model输出的是维度: [batch_size, action_dim, atoms]是一个概率矩阵
                首先是上面计算q_vals, 用model的输出乘以support再在atoms维度上求和得到每个action对应的q_vals
                然后选择q_val最大的action, max操作过后的维度是: [batch_size]
                接着reshape和expand, 因为后面需要用到这个action，选择target_model里的概率，
                所以必须整理成: [batch_size, 1, atoms]
                '''
                # project the bellman update
                '''
                
                '''
                next_distribution_probs = self.target_model(obses_tp1).gather(1, actions_tp1)    # [batch_size, 1, atoms]
                next_distribution_probs = next_distribution_probs.squeeze()   # [batch_size, atoms]
                
                # Tz = rewards + self.gamma**self.n_steps * (self.target_model(obses_tp1) * self.supports) * dones # shape: [batch_size, action_dim, atoms]
                # Tz = Tz.gather(1, actions_tp1) # shape: [batch_size, 1, atoms]  # torch.gather(dim, index): Index tensor must have the same number of dimensions as input tensor
                '''
                一开始这里关于Tz的计算公式写错了，Tz = r + gamma * supports，应该是直接用supports，而我写成了next_distribution_probs*supports
                '''
                Tz = rewards + self.gamma**self.n_steps * self.supports * (1 - dones)   # [batch_size, 1, atoms]  是1 - dones不是done！！！！！！！！！！
                Tz = Tz.clamp(self.v_min, self.v_max)
                Tz = Tz.squeeze()     # [batch_size, atoms]
                
                b = (Tz - self.v_min) / self.delta_z  # 分在第几块里 [batch_size, 1, atoms]  b in [0, self.atom-1]
                l = b.floor()
                u = b.ceil()
                # 处理恰好落在边界的情况
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1
                
                # ?Tz.index_add_
                '''
                这个投影不是那么好写的，一开始想当然了写成 m = next_distribution_probs * (u - b) + next_distribution_probs * (b - l) # [batch_size, 1, atoms]
                而且分别的概率应该是与距离成反比，即距离越近，分配的概率应该越大!!!
                '''
                m = torch.zeros_like(Tz)
                l_p = next_distribution_probs * (u - b)
                u_p = next_distribution_probs * (b - l)
                
                for i in range(m.shape[0]):
                    m[i].index_add_(0, l[i].to(torch.int64), l_p[i])
                    m[i].index_add_(0, u[i].to(torch.int64), u_p[i])
    
                '''
                这种解法更高效，但是我还没有完全理解
                '''
                # offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.atoms)
                # m = torch.zeros(self.batch_size, self.atoms)
                # m.view(-1).index_add_(0, (l + offset).view(-1).to(torch.int64), (next_distribution_probs * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
                # m.view(-1).index_add_(0, (u + offset).view(-1).to(torch.int64), (next_distribution_probs * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
                # print(m.shape)
    
    
    
            # compute loss
            current_distribution_probs = self.model(obses_t).gather(1, actions).squeeze()  # [batch_size, atoms]
            
            loss = -(m * current_distribution_probs.log()).sum(dim=-1)
        
            if self.prioritized:
                loss = loss * torch.tensor(weights)
                self.memory.update_priorities(indices, loss.detach().squeeze().abs().numpy().tolist())
            
            loss = loss.mean()
            
        
            
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():      # 梯度裁剪非常有用！！！
                param.grad.data.clamp(-1, 1)
            self.optimizer.step()
                   
            self.losses.append(loss.item())
            
            # update target model
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            
            # 这一段可以放在training loop里
            # if self.update_count % 10000 == 0:
            #     mean_returns = self.eval(5)
            #     if mean_returns > 300:
            #         self.render()
                
        
        def eval_(self, n_trajs):
            env = gym.make(self.env_name)
            self.eval = True
            episode_return = 0
            episode_length = 0            
            for _ in range(n_trajs):
                obs = env.reset()
                for _ in range(1000):
                    a = self.get_action(obs)
                    obs, reward, done, info = env.step(a)
                    episode_return += reward
                    episode_length += 1
                    
                    if done:
                        self.rewards.append(episode_return)
                        self.episode_length.append(episode_length)
                        episode_return = 0
                        episode_length = 0
                        break
                        
            # print('eval {} trajs, mean return: {}'.format(n_trajs, np.mean(episode_returns)))
            env.close()
            self.eval = False
            return np.mean(self.rewards[-n_trajs:]), np.max(self.rewards[-n_trajs:]), np.mean(self.episode_length[-n_trajs:]), np.max(self.episode_length[-n_trajs:])
        

                
        def get_action(self, obs, eps=0.1):   # epsilon-greedy policy
            with torch.no_grad():
                if np.random.random() >= eps or self.eval_mode:
                    # print(s.dtype)
                    obs = np.expand_dims(obs, 0)
                    obs = torch.tensor(obs, dtype=torch.float)
                    next_distribution = self.model(obs) * self.supports
                    q_vals = next_distribution.sum(dim=-1)
                    a = q_vals.max(dim=1)[1]
                    return a.item()
                else:
                    return np.random.randint(0, self.action_dim)
        
    
        def n_steps_replay(self, transition):
            '''
            如果想要使用 n-steps TD learning,  在收集transition时就用这个函数，而不是用self.memory.add()
            模拟证明还是挺有用的，在LunarLander-v2上，其他DQN tricks都不加，只是用n-steps=5,就能训练得蛮好的，
            虽然前期的loss趋势很诡异，会先上升一段，然后开始慢慢下降 (可能是由于当时计算target的时候V(s')的折现只考虑了一步，已经改正）
            '''
            _, _, _, obs_tpn, done = transition
            self.n_step_buffer.append(transition)
            
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            R = sum([self.n_step_buffer[i][2] * self.gamma**i for i in range(self.n_steps)])
            obs_t, action, _, _, _ = self.n_step_buffer.pop(0)
            
            self.memory.add((obs_t, action, R, obs_tpn, done))
        
        
        
        def save_w(self):
            # Returns a dictionary containing a whole state of the module.
            torch.save(self.model.state_dict(), './model.dump')
            torch.save(self.optimizer.state_dict(), './optim.dump')
            
        def load_w(self):
            fname_model = './model.dump'
            fname_optim = './optim.dump'
            
            if os.path.isfile(fname_model):
                self.model.load_state_dict(torch.load(fname_model))
                self.target_model.load_state_dict(self.model.state_dict())
                
            if os.path.isfile(fname_optim):
                self.optimizer.load_state_dict(torch.load(fname_optim))
                
        def save_replay(self):
            pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
            
        def load_replay(self):
            fname = './exp_replay_agent.dump'
            if os.path.isfile(fname):
                self.memory = pickle.load(open(fname, 'rb'))
                
                
        def huber_loss(self, x, delta):    # x must be tensor
            cond = (x.abs() <= delta).to(torch.float32)
            return 0.5 * x.pow(2) * cond + delta * (x.abs() - 0.5 * delta) * (1 - cond)
        
        
        def render(self):
            env = gym.make(self.env_name)
            self.eval = True
            obs = env.reset()
            for _ in range(1000):
                env.render()
                a = self.get_action(obs)
                obs, reward, done, info = env.step(a)
                if done:
                    break
            env.close()
            self.eval = False










class QuantileDQNAgentPytorch(object):    
    
        def __init__(self, env_name=None, network=QuantileNetwork_pytorch, 
                     quantiles = 51,
                     double=False, prioritized=False, n_steps=1, eval_mode=False, config=config):
            
            self.env_name = env_name
            self.env = gym.make(env_name)
            self.obs_dim = self.env.observation_space.shape[0]   # 根据环境来设置
            self.action_dim = self.env.action_space.n
            self.eval_mode = eval_mode
            
            
            self.quantiles = quantiles
            # self.cumulative_density = torch.tensor((2 * np.arange(self.quantiles) + 1) / (2.0 * self.quantiles))  # 注意这个设置分位点的方式，tau_hat，参考论文的lemma 2
            
            
            
            self.gamma = config.gamma
            self.lr = config.lr
            self.target_net_update_freq = config.target_net_update_freq
            self.experience_replay_size = config.exp_replay_size
            self.batch_size = config.batch_size
            self.learn_start = config.learn_start     
            
            self.prioritized = prioritized
            self.alpha = config.alpha
            self.beta_start = config.beta_start
            self.beta_steps = config.beta_steps
            self.n_steps = n_steps
            self.n_step_buffer = []
            
            if self.prioritized:
                self.memory = PrioritizedReplayMemory(self.experience_replay_size, self.alpha, self.beta_start, self.beta_steps)
            else:
                self.memory = ExperienceReplayMemory(self.experience_replay_size)
            
            
            self.network = network
            self.double = double
            
    
            self.model = self.network(self.obs_dim, self.action_dim, self.quantiles)
            self.target_model = self.network(self.obs_dim, self.action_dim, self.quantiles)
            self.target_model.load_state_dict(self.model.state_dict())
            
            # train和eval模式的差别主要是Batch Normalization和Dropout的使用差别
            if self.eval_mode:
                self.model.eval()
                self.target_model.eval()
            else:
                self.model.train()
                self.target_model.train()
            
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
            # loss 自己在下面写
                    
            self.update_count = 0
            self.losses = []
            self.rewards = []
            self.episode_length = []
            
            
        
        def train(self, step=0):
            if self.eval_mode:
                return None
            
            if step < self.learn_start:
                return None
            
            if self.prioritized:
                transitions, indices, weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
            
            
            # if self.prioritized:
            #     obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            # else:
            #     obses_t, actions, rewards, obses_tp1, dones = transitions
            obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            
            obses_t = torch.tensor(obses_t, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long).squeeze().view(-1, 1, 1).expand(-1, -1, self.quantiles)
            rewards = torch.tensor(rewards, dtype=torch.float).squeeze().view(-1, 1, 1)
            obses_tp1 = torch.tensor(obses_tp1, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.int32).squeeze().view(-1, 1, 1)  
                        
            
            #============= compute loss ===================   
            current_quantiles = self.model(obses_t).gather(1, actions).squeeze()  # [batch_size, quantiles]
            
            with torch.no_grad():      
                if self.double:
                    actions_tp1 = self.model(obses_tp1).mean(dim=-1).max(dim=1)[1].view(-1, 1, 1).expand(-1, -1, self.quantiles)  # [batch_size, 1, quantiles]
                    next_quantiles = self.target_model(obses_tp1).gather(1, actions_tp1)   # [batch_size, 1, quantiles]
                else:
                    actions_tp1 = self.target_model(obses_tp1).mean(dim=-1).max(dim=1)[1].view(-1, 1, 1).expand(-1, -1, self.quantiles)   # [batch_size, 1, quantiles]
                    next_quantiles = self.target_model(obses_tp1).gather(1, actions_tp1)    # [batch_size, 1, quantiles]
            
                # compute target

                targets = rewards + self.gamma**self.n_steps * next_quantiles * (1 - dones) # [batch_size, 1, quantiles]
                targets = targets.squeeze()    # [batch_size, quantiles]
            
            
            bellman_errors = targets[:, None, :] - current_quantiles[:, :, None]    # [batch_size, quantiles, quantiles]
            huber_loss = self.huber_loss(bellman_errors)
            
            tau_hat = (torch.arange(self.quantiles, dtype=torch.float32) + 0.5) / self.quantiles  # [quantiles]
            
            quantile_huber_loss = (tau_hat - (bellman_errors < 0).to(torch.float)).abs() * huber_loss
            loss = quantile_huber_loss.mean(dim=2).sum(dim=1)    # [batch_size]
            
            
            
            if self.prioritized:
                loss = loss * torch.tensor(weights)
                self.memory.update_priorities(indices, loss.detach().squeeze().abs().numpy().tolist())
            
            loss = loss.mean()
            
        
            
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():      # 梯度裁剪非常有用！！！
                param.grad.data.clamp(-1, 1)
            self.optimizer.step()
                   
            self.losses.append(loss.item())
            
            # update target model
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            
            # 这一段可以放在training loop里
            # if self.update_count % 10000 == 0:
            #     mean_returns = self.eval(5)
            #     if mean_returns > 300:
            #         self.render()
                
        
        def eval_(self, n_trajs):
            env = gym.make(self.env_name)
            self.eval = True
            episode_return = 0
            episode_length = 0            
            for _ in range(n_trajs):
                obs = env.reset()
                for _ in range(1000):
                    a = self.get_action(obs)
                    obs, reward, done, info = env.step(a)
                    episode_return += reward
                    episode_length += 1
                    
                    if done:
                        self.rewards.append(episode_return)
                        self.episode_length.append(episode_length)
                        episode_return = 0
                        episode_length = 0
                        break
                        
            # print('eval {} trajs, mean return: {}'.format(n_trajs, np.mean(episode_returns)))
            env.close()
            self.eval = False
            return np.mean(self.rewards[-n_trajs:]), np.max(self.rewards[-n_trajs:]), np.mean(self.episode_length[-n_trajs:]), np.max(self.episode_length[-n_trajs:])
        

                
        def get_action(self, obs, eps=0.1):   # epsilon-greedy policy
            with torch.no_grad():
                if np.random.random() >= eps or self.eval_mode:
                    # print(s.dtype)
                    obs = np.expand_dims(obs, 0)
                    obs = torch.tensor(obs, dtype=torch.float)
                    q_vals = self.model(obs).mean(dim=-1)
                    a = q_vals.max(dim=1)[1]            
                    return a.item() 
                else:
                    return np.random.randint(0, self.action_dim)
        
    
        def n_steps_replay(self, transition):
            '''
            如果想要使用 n-steps TD learning,  在收集transition时就用这个函数，而不是用self.memory.add()
            模拟证明还是挺有用的，在LunarLander-v2上，其他DQN tricks都不加，只是用n-steps=5,就能训练得蛮好的，
            虽然前期的loss趋势很诡异，会先上升一段，然后开始慢慢下降 (可能是由于当时计算target的时候V(s')的折现只考虑了一步，已经改正）
            '''
            _, _, _, obs_tpn, done = transition
            self.n_step_buffer.append(transition)
            
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            R = sum([self.n_step_buffer[i][2] * self.gamma**i for i in range(self.n_steps)])
            obs_t, action, _, _, _ = self.n_step_buffer.pop(0)
            
            self.memory.add((obs_t, action, R, obs_tpn, done))
        
        
        
        def save_w(self):
            # Returns a dictionary containing a whole state of the module.
            torch.save(self.model.state_dict(), './model.dump')
            torch.save(self.optimizer.state_dict(), './optim.dump')
            
        def load_w(self):
            fname_model = './model.dump'
            fname_optim = './optim.dump'
            
            if os.path.isfile(fname_model):
                self.model.load_state_dict(torch.load(fname_model))
                self.target_model.load_state_dict(self.model.state_dict())
                
            if os.path.isfile(fname_optim):
                self.optimizer.load_state_dict(torch.load(fname_optim))
                
        def save_replay(self):
            pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
            
        def load_replay(self):
            fname = './exp_replay_agent.dump'
            if os.path.isfile(fname):
                self.memory = pickle.load(open(fname, 'rb'))
                
                
        def huber_loss(self, x, delta=1):    # x must be tensor
            cond = (x.abs() <= delta).to(torch.float32)
            return 0.5 * x.pow(2) * cond + delta * (x.abs() - 0.5 * delta) * (1 - cond)
        
        
        def render(self):
            env = gym.make(self.env_name)
            self.eval = True
            obs = env.reset()
            for _ in range(1000):
                env.render()
                a = self.get_action(obs)
                obs, reward, done, info = env.step(a)
                if done:
                    break
            env.close()
            self.eval = False


        
