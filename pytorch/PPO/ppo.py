# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:09:04 2023

@author: leyuan

reference:
    https://github.com/openai/spinningup/tree/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo
"""

import os
import numpy as np
import scipy.signal


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super(MLPCategoricalActor, self).__init__()
        
        self.logits_net = mlp([obs_dim]+list(hidden_sizes)+[action_dim], activation)
        
        
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, action):
        
        return pi.log_prob(action)
    
    
    def forward(self, obs, action=None):
       # Produce action distributions for given observations, and 
       # optionally compute the log likelihood of given actions under
       # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(pi, action)
        return pi, logp_a
    

    
    
    
class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super(MLPGaussianActor, self).__init__()
        
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim]+list(hidden_sizes)+[action_dim], activation)
        
        
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, action):
        
        return pi.log_prob(action).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def forward(self, obs, action=None):
        pi = self._distribution(obs)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(pi, action)
        return pi, logp_a    




class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        
        self.v_net = mlp([obs_dim]+list(hidden_sizes)+[1], activation)
        
    def forward(self, obs):
        value = self.v_net(obs)
        
        return torch.squeeze(value, dim=-1)   # remove size '1' dim.
    
    

    


class PPOBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, num_episodes=5, use_gae=True, gamma=0.99, lamda=0.95):
        self.obses = []
        self.actions = []
        self.rewards = []
        # self.dones = []
        self.log_probs = []
        self.values = [] 
        self.returns = []    # reward-to-go
        self.advantages = []
        
        self.capacity = num_episodes
        
        self.use_gae = use_gae
        self.lamda = lamda           # param for gae
        self.gamma = gamma           # discount factor
        
        
        self.ptr = 0                 # current position
        self.path_start_idx = 0      # the position of the start of a trajectory
        self.traj_idx = 0            # indicate the i-th traj
 



    def add(self, obs, action, reward, log_p, value):
        assert self.traj_idx < self.capacity
        
        self.obses.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_p)
        self.values.append(value)
        
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory. This looks back in the buffer to where the trajectory started, 
        and uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as the targets for the value function.
        
        The "last_val" argument should be 0 if the trajectory terminated because the agent reached a terminal state (died), 
        and otherwise (truncated) should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(np.array(self.rewards[path_slice]), last_val)
        values = np.append(np.array(self.values[path_slice]), last_val)   # np.array(self.values[path_slice].append(last_val)) 这种写法得到的是None
        
        # calculate advantage estimates
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        if self.use_gae:            
            self.advantages += list(discount_cumsum(deltas, self.gamma * self.lamda))
        else:
            # 使用TD(1), 也可以使用MC估计，也就是计算reward-to-go
            # self.advantages += list(deltas)   # TD(1)在HalfCheetah上不行
            self.advantages += list(discount_cumsum(rewards, self.gamma)[:-1])
            
        # compute reward-to-go, to be targets for the value function update
        self.returns += list(discount_cumsum(rewards, self.gamma)[:-1])
        
        self.path_start_idx = self.ptr
        self.traj_idx += 1
        
        
    def output(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer and clear the buffer.
        """
        assert self.traj_idx == self.capacity   # buffer has to be full before output
        self.ptr, self.path_start_idx, self.traj_idx = 0, 0, 0
        
        
        data = dict(obs=self.obses, act=self.actions, ret=self.returns, adv=self.advantages, logp=self.log_probs)        
        data = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        # advantage normalization trick
        data['adv'] = (data['adv'] - data['adv'].mean()) / data['adv'].std()
        
        # clear buffer
        del self.obses[:]
        del self.actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:] 
        del self.returns[:]
        del self.advantages[:]
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
    

'''
class Config(object):
    def __init__(self):
        pass

config = Config()
config.capacity = 4              # capacity of the buffer (the number of episodes in the buffer).
config.epochs = 50               # Number of epochs of interaction (equivalent to number of policy updates) to perform.
config.gamma = 0.99              # Discount factor.
config.clip_ratio = 0.2          # Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. (Usually small, 0.1 to 0.3.)
config.actor_lr = 3e-4           # Learning rate for policy optimizer.
config.critic_lr = 1e-3          # Learning rate for value function optimizer.
config.train_actor_iters = 80    # Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)
config.train_critic_iters = 80   # Number of gradient descent steps to take on value function per epoch.
config.lamda = 0.97              # Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
config.max_ep_len = 1000         # Maximum length of trajectory / episode / rollout.
config.target_kl = 0.01          # Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
config.save_freq = 10            # How often (in terms of gap between epochs) to save the current policy and value function.
config.actor_hiddens = (64, 64)  # network architecture for actor.
config.critic_hiddens = (64, 64) # network architecture for critic.
config.activation = nn.Tanh      # activation function for actor and critic.
config.use_gae = True            # whether use GAE-lambda for advantage estimation
'''   


class PPO(object):
    def __init__(self, obs_dim, action_dim, is_discrete=False, config=None):
        
        # Building actor
        if is_discrete:
            self.actor = MLPCategoricalActor(obs_dim, action_dim, config.actor_hiddens, config.activation)
        else:
            self.actor = MLPGaussianActor(obs_dim, action_dim, config.actor_hiddens, config.activation)
        
        # Building critic
        self.critic = MLPCritic(obs_dim, config.critic_hiddens, config.activation)    
        
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        self.actor_losses = []
        self.critic_losses = []
                
        self.config = config
        
    def step(self, obs):
        with torch.no_grad():
            pi = self.actor._distribution(obs)
            a = pi.sample()
            logp_a = self.actor._log_prob_from_distribution(pi, a)
            v = self.critic(obs)
        
        return a.numpy(), v.numpy(), logp_a.numpy()
    
    
    def get_action(self, obs):
        
        return self.step(obs)[0]
        
    
    def compute_actor_loss(self, data):
        obses, actions, advantages, log_probs_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss
        pis, log_probs = self.actor(obses, actions)
        ratios = torch.exp(log_probs - log_probs_old)
        advantage_clipped = torch.clamp(ratios, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        actor_loss = -(torch.min(ratios * advantages, advantage_clipped)).mean()
        
    
        # Useful extra info
        approx_kl = (log_probs_old - log_probs).mean().item()   # KL(pi_old, pi)
        ent = pis.entropy().mean().item()
        is_clipped = ratios.gt(1 + self.config.clip_ratio) | ratios.lt(1 - self.config.clip_ratio)
        clipfrac = torch.as_tensor(is_clipped, dtype=torch.float32).mean().item()   # clip的比例
        pi_info = dict(kl=approx_kl, entropy=ent, clip_ratio=clipfrac)
        
        
        return actor_loss, pi_info
    
    
    
    def compute_critic_loss(self, data):
        obses, returns = data['obs'], data['ret']        
        critic_loss = self.critic(obses) - returns
        
        return (critic_loss**2).mean()
    
    

    def update(self, buffer, epoch):
        data = buffer.output()
        
        # policy update
        for i in range(self.config.train_actor_iters):
            self.actor_optimizer.zero_grad()
            
            actor_loss, pi_info = self.compute_actor_loss(data)
            self.actor_losses.append(actor_loss.item())
            kl = pi_info['kl']
            if kl > 1.5 * self.config.target_kl:
                print(f'Early stopping at step {i} due to reaching max kl.')
                break
            
            actor_loss.backward()
            self.actor_optimizer.step()
            
            
        # value update
        for j in range(self.config.train_critic_iters):
            self.critic_optimizer.zero_grad()
            critic_loss = self.compute_critic_loss(data) 
            self.critic_losses.append(critic_loss.item())
            critic_loss.backward()
            self.critic_optimizer.step()
            
            
        print(f'============== epoch: {epoch} ================')
        print(f"actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}.")
        print(f"approx kl: {pi_info['kl']}, entropy: {pi_info['entropy']}, clip ratio: {pi_info['clip_ratio']}.")
        
        
        
    def save(self, step):
        torch.save(self.actor.state_dict(), './ppo_actor_{}.pkl'.format(step))
        torch.save(self.critic.state_dict(), './ppo_critic_{}.pkl'.format(step))

        
    def load(self, path_actor, path_critic):
        if os.path.isfile(path_actor):
            self.critic.load_state_dict(torch.load(path_actor))
            # self.policy.load_state_dict(torch.load(path), map_location=lambda storage, loc: storage))  # 在gpu上训练，load到cpu上的时候可能会用到
        else:
            print('No "{}" exits for loading'.format(path_actor)) 
        if os.path.isfile(path_critic):
            self.critic.load_state_dict(torch.load(path_critic))
            # self.policy.load_state_dict(torch.load(path), map_location=lambda storage, loc: storage))  # 在gpu上训练，load到cpu上的时候可能会用到
        else:
            print('No "{}" exits for loading'.format(path_critic))      
        
            









