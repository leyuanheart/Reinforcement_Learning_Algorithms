# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:27:31 2023

@author: leyuan

reference:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
    https://github.com/pranz24/pytorch-soft-actor-critic     这里面有automatic_entropy_tuning
    
这里还是使用固定的Entropy约束。
"""



import numpy as np
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal



class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, capacity=int(1e6)):
        '''
        如果只在__init__()中用到的变量就没必要用self.var来多表示一遍了
        '''
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, action_dim))
        self.next_obs = np.zeros((capacity, obs_dim))
        self.reward = np.zeros((capacity, ))
        self.done = np.zeros((capacity, ))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        
    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        
        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
            )



LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]       
        
    return nn.Sequential(*layers)



class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, action_scale, action_bias):
        super(Actor, self).__init__()
        
        self.net = mlp([obs_dim]+list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        
        self.action_scale = action_scale
        self.action_bias = action_bias
        
        
    def forward(self, obs, deterministic=False, with_logp=True):
       
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Pre-squash distribution and sample
        pi = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            u = mu
        else:
            u = pi.rsample() # for reparameterization trick 
        
        # Enforcing action bounds
        a = torch.tanh(u) 
        if with_logp:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.

            # log_prob = (pi.log_prob(u) - torch.log(self.action_scale * (1 - a**2) + 1e-7)).sum(dim=-1)   # Eq 21
            log_prob = (pi.log_prob(u) - (np.log(self.action_scale) + 2 * (np.log(2) - u - F.softplus(-2 * u)))).sum(axis=1)
        else:
            log_prob = None
    
        return a * self.action_scale + self.action_bias, log_prob
    
    
    
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super(Critic, self).__init__()
        
        self.q1 = mlp([obs_dim+action_dim]+list(hidden_sizes)+[1], activation)
        self.q2 = mlp([obs_dim+action_dim]+list(hidden_sizes)+[1], activation)
        
    def forward(self, obs, action):
        oa = torch.cat([obs, action], dim=-1)
        q1 = self.q1(oa)
        q2 = self.q2(oa)
        
        return torch.squeeze(q1, -1), torch.squeeze(q2, -1) # Critical to ensure q has right shape.return 
    
    def Q1(self, obs, action):
        oa = torch.cat([obs, action], dim=-1)
        q1 = self.q1(oa)
        
        return torch.squeeze(q1, -1)


'''
class Config(object):
    def __init__(self):
        pass

config = Config()
config.capacity = int(1e6)       # Maximum size of replay buffer.
config.gamma = 0.99              # Discount factor.
# config.max_ep_len = 1000         # Maximum length of trajectory / episode / rollout.
config.alpha = 0.2               # Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)


config.actor_hiddens = (64, 64)  # network architecture for actor.
config.critic_hiddens = (64, 64) # network architecture for critic.
config.activation = nn.ReLU      # activation function for actor and critic.
config.rho = 0.995               # polyak averaging hyperparm for target network update: \theta_{targ} \leftarrow \rho \theta_{targ} + (1-\rho) \theta
config.actor_lr = 1e-3           # Learning rate for actor optimizer.
config.critic_lr = 1e-3          # Learning rate for critic optimizer.


config.batch_size = 64           # Minibatch size for SGD.
config.max_timesteps = int(5e4)  # Maximum number of steps of interaction (state-action pairs) for the agent and the environment.
config.start_timesteps = 10000       # Number of steps for uniform-random action getion, before running real policy. Helps exploration.
config.update_after = 1000       # Number of env interactions to collect before starting to do gradient descent updates. Ensures replay buffer is full enough for useful updates.
config.update_every = 1000       # Number of env interactions that should elapse between gradient descent updates. Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1. 其实设置成1就可以了。
config.eval_freq = 5000          # How often (time steps) we evaluate land save the model.
'''


class SAC(object):
    def __init__(self, obs_dim, action_dim, action_scale, action_bias, config=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(obs_dim, action_dim, config.actor_hiddens, config.activation, action_scale, action_bias).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)  # lr=1e-4
        
        
        self.critic = Critic(obs_dim, action_dim, config.critic_hiddens, config.activation).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)  # weight_decay=1e-2, weight_decay就是在目标函数中加上参数的二范数惩罚项
        
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.gamma = config.gamma
        self.rho = config.rho
        self.alpha = config.alpha
           
    
        self.actor_losses = []
        self.critic_losses = []
        
        self.total_it = 0
        
    def get_action(self, obs, deterministic):
        # interact with env
        with torch.no_grad():            
            action, _ = self.actor(obs, deterministic, with_logp=False)
        
        return action.numpy()
    
    
    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        # sample from replay buffer
        obs, action, next_obs, reward, done = replay_buffer.sample(batch_size)
        
        # Q-network update
        with torch.no_grad():
            next_action, next_log_p = self.actor(next_obs)   # 这里应该用的是actor，不是actor_target
            
            # compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_p
            target_Q = reward + self.gamma * (1 - done) * target_Q
                
        # get current Q estimate
        current_Q1, current_Q2 = self.critic(obs, action)
        
        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_losses.append(critic_loss.item())
        
        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update the frozen target models (soft update)
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                # target_param.data.copy_(self.rho * target_param.data + (1 - self.rho) * param.data)
                target_param.data.mul_(self.rho)
                target_param.data.add_((1 - self.rho) * param.data)
            
        
        # policy update
            
        # Freeze Q-network so you don't waste computational effort computing gradients for it during the actor learning step.
        for p in self.critic.parameters():
            p.requires_grad = False
            
        # compute actor loss
        a, log_p = self.actor(obs)
        q1, q2 = self.critic(obs, a)
        q = torch.min(q1, q2)   # 也可以试试只用Q1
        actor_loss = (- q + self.alpha * log_p).mean()
        self.actor_losses.append(actor_loss.item())
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Unfreeze Q-network so you can optimize it at next TD3 step.
        for p in self.critic.parameters():
            p.requires_grad = True
        
              
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        


# def eval_policy(env_name, agent, seed, eval_episodes=5):
#     env = gym.make(env_name)
        
#     avg_return = 0.
#     for _ in range(eval_episodes):
#         obs, info = env.reset(seed=seed)
#         done = False
#         while not done:
#             action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             avg_return += reward
#             done = terminated or truncated
            
#     avg_return /= eval_episodes
    
#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes, average return: {avg_return:.3f}")
#     print("---------------------------------------")
#     env.close()
#     return avg_return




# def render(env_name, agent, seed=None):
#     env = gym.make(env_name, render_mode='human')
#     obs, info = env.reset(seed=seed)

#     returns = 0
#     for i in range(1000):
#         action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
#         obs, reward, terminated, truncated, _ = env.step(action)
#         returns += reward
#         done = terminated or truncated
#         if done:
#             print(returns)
#             break
#     env.close()
 

# eval_policy(env_name, sac_agent, seed)        
# render(env_name, sac_agent)       

# if __name__ == '__main__':
#     main()







