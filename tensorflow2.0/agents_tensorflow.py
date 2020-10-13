#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:01:42 2020

@author: leyuan
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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.list_physical_devices(device_type='GPU')


import gym


from configs import Config
from replay_memories import ExperienceReplayMemory, PrioritizedReplayMemory
from networks_tensorflow import MLP_tensorflow, CONV_tensorflow, QuantileNetwork_tensorflow

config = Config()



class DQNAgentTensorflow(object):
        def __init__(self, env_name=None, network=MLP_tensorflow, double=False, prioritized=False, n_steps=1, eval_mode=False, config=None):
            
            self.env_name = env_name
            self.env = gym.make(self.env_name)
            self.env.seed(config.training_env_seed)
            
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
                
            # Tensorflow的train和eval模式是通过__call__()的training参数来设置的
                
            
            self.network = network
            self.double = double
    
            self.model = self.network(self.action_dim)
            self.target_model = self.network(self.action_dim)
            self.target_model.set_weights(self.model.get_weights())
            
            
            exponential_decay = optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=10000, decay_rate=1)
            self.optimizer = optimizers.Adam(learning_rate=exponential_decay)
            # self.optimizer = optimizers.Adam(lr=self.lr)
            self.loss = tf.losses.Huber(delta=1., reduction='none')
                    
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
                transitions, replay_buffer_indices, weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
                
            obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            
            obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=tf.int32))
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            obses_tp1 = tf.convert_to_tensor(obses_tp1, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                # Q(s,a)
                q_vals = self.model(obses_t)
                indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
                chosen_q_vals = tf.gather_nd(q_vals, indices=indices)
                # print('q_vals shape:{}, chosen_q_vals shape:{}'.format(q_vals.shape, chosen_q_vals.shape))
                if self.double:
                    actions_tp1 = tf.cast(tf.squeeze(tf.argmax(self.model(obses_tp1), axis=-1)), dtype=tf.int32)
                    indices = tf.stack([tf.range(actions_tp1.shape[0]), actions_tp1], axis=-1)   # 一开始这里tf.stack里写的是actions, 观察训练loss不收敛，然后发现了错误
                    q_tp1_vals = tf.gather_nd(self.target_model(obses_tp1), indices=indices)
                else:
                    q_tp1_vals = tf.math.reduce_max(self.target_model(obses_tp1), axis=-1)
                
                targets = tf.stop_gradient(rewards + self.gamma**self.n_steps * q_tp1_vals * (1-dones))
                
                if self.prioritized:
                    loss = self.loss(chosen_q_vals, targets) * tf.constant(weights, dtype=tf.float32)
                    diff = chosen_q_vals - targets
                    self.memory.update_priorities(replay_buffer_indices, np.abs(tf.squeeze(diff).numpy()).tolist())
                else:
                    loss = self.loss(chosen_q_vals, targets)
                
                
                loss = tf.reduce_mean(loss)
                
            self.grads = tape.gradient(loss, self.model.trainable_variables)
            # self.grads = [tf.clip_by_value(grad, -1, 1) for grad in grads]
        
            self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_variables))
            
            self.losses.append(loss.numpy())
            
            # update target model
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            
            '''
            也可以使用soft update: traget_weight = (1 - 0.999) * model_weight + 0.999 * target_weight
            '''
            # params = self.model.get_weights()
            # target_params = self.target_model.get_weights()
            # for idx in range(len(params)):
            #     target_params[idx] = (1 - 0.999) * params[idx] + 0.999 * target_params[idx]
            # self.target_model.set_weights(target_params)
            
        
        def get_action(self, obs, eps=0.1, training=None):   # epsilon-greedy policy
            if np.random.random() >= eps or self.eval_mode:
                # print(s.dtype)
                obs = np.expand_dims(obs, 0)
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                a = tf.stop_gradient(tf.argmax(self.model(obs, training=training), axis=-1))
                return a.numpy()[0]
            else:
                return np.random.randint(0, self.action_dim)    
        
        
        def eval_(self, env, n_trajs):
        
            self.eval_mode = True
            
            for _ in range(n_trajs):
                episode_return = 0
                episode_length = 0
                obs = env.reset()
                for _ in range(10000):
                    a = self.get_action(obs, training=self.eval_mode)
                    obs, reward, done, info = env.step(a)
                    episode_return += reward
                    episode_length += 1
                    
                    if done:
                        self.rewards.append(episode_return)
                        self.episode_length.append(episode_length)
                     
                        break
                        
            # print('eval {} trajs, mean return: {}'.format(n_trajs, np.mean(episode_returns)))
            
            self.eval_mode = False
            return np.mean(self.rewards[-n_trajs:]), np.max(self.rewards[-n_trajs:]), np.mean(self.episode_length[-n_trajs:]), np.max(self.episode_length[-n_trajs:])
        
        
        
        
        
        def n_steps_replay(self, transition):
            '''
            如果想要使用 n-steps TD learning,  在收集transition时就用这个函数，而不是用self.memory.add()
            模拟证明还是挺有用的，在LunarLander-v2上，其他DQN tricks都不加，只是用n-steps=5,就能训练得蛮好的，
            虽然前期的loss趋势很诡异，会先上升一段，然后开始慢慢下降
            '''
            _, _, _, obs_tpn, done = transition
            self.n_step_buffer.append(transition)
            
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            R = sum([self.n_step_buffer[i][2] * self.gamma**i for i in range(self.n_steps)])
            obs_t, action, _, _, _ = self.n_step_buffer.pop(0)
            
            self.memory.add((obs_t, action, R, obs_tpn, done))
        
        
        def save_w(self):
            self.model.save_weights('./model_weights.ckpt')
    
    
        def load_w(self):
            fname_model = './model_weights.ckpt'
            
            if os.path.isfile(fname_model):
                self.model.load_weights(fname_model)
                
                
        def save_replay(self):
            pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
            
        def load_replay(self):
            fname = './exp_replay_agent.dump'
            if os.path.isfile(fname):
                self.memory = pickle.load(open(fname, 'rb'))
                
                
        def render(self, env):
            
            
            self.eval_mode = True
            obs = env.reset()
            for _ in range(10000):
                env.render()
                a = self.get_action(obs, training=self.eval_mode)
                obs, reward, done, info = env.step(a)
                if done:
                    break

            self.eval_mode = False
            




class QuantileDQNAgentTensorflow(object):
        def __init__(self, env_name=None, network=QuantileNetwork_tensorflow, 
                     quantiles = 51, 
                     double=False, prioritized=False, n_steps=1, eval_mode=False, config=None):
            
            self.env_name = env_name
            self.env = gym.make(self.env_name)
            self.env.seed(config.training_env_seed)
            
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
                
            # Tensorflow的train和eval模式是通过__call__()的training参数来设置的
                
            
            self.network = network
            self.double = double
            
            self.quantiles = quantiles
    
            self.model = self.network(self.action_dim, self.quantiles)
            self.target_model = self.network(self.action_dim, self.quantiles)
            self.target_model.set_weights(self.model.get_weights())
                
            
            exponential_decay = optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=10000, decay_rate=1)
            self.optimizer = optimizers.Adam(learning_rate=exponential_decay)
            # self.optimizer = optimizers.Adam(lr=self.lr)
            self.huber_loss = tf.losses.Huber(delta=1., reduction='none')
                    
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
                transitions, replay_buffer_indices, weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
                
            obses_t, actions, rewards, obses_tp1, dones = zip(*transitions)
            
            obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
            actions = tf.squeeze(tf.convert_to_tensor(actions, dtype=tf.int32))
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            obses_tp1 = tf.convert_to_tensor(obses_tp1, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                # Q(s,a)
                current_quantiles = self.model(obses_t)                             # [batch_size, action_dim, quantiles]
                # current_q = tf.reduce_mean(current_quantiles, axis=-1)            # [batch_size, action_dim]
                indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)  # [batch_size, 2]
                current_quantiles = tf.gather_nd(current_quantiles, indices=indices)            # [batch_size, quantiles]
                # print('q_vals shape:{}, chosen_q_vals shape:{}'.format(q_vals.shape, chosen_q_vals.shape))
                if self.double:
                    next_model_quantiles = self.model(obses_tp1)                           # [batch_size, action_dim, quantiles] 
                    next_model_q = tf.reduce_mean(next_model_quantiles, axis=-1)                 # [batch_size, action_dim]
                    actions_tp1 = tf.cast(tf.squeeze(tf.argmax(next_model_q, axis=-1)), dtype=tf.int32)   # [batch_size]
                    indices = tf.stack([tf.range(actions_tp1.shape[0]), actions_tp1], axis=-1)      # [batch_size, 2]
                    next_target_quantiles = self.target_model(obses_tp1)                           # [batch_size, action_dim, quantiles] 
                    next_quantiles = tf.gather_nd(next_target_quantiles, indices=indices)                  # [batch_size, quantiles]
                else:                
                    next_quantiles = self.target_model(obses_tp1)                           # [batch_size, action_dim, quantiles] 
                    next_q = tf.reduce_mean(next_quantiles, axis=-1)                 # [batch_size, action_dim]
                    actions_tp1 = tf.cast(tf.squeeze(tf.argmax(next_q, axis=-1)), dtype=tf.int32)   # [batch_size]
                    indices = tf.stack([tf.range(actions_tp1.shape[0]), actions_tp1], axis=-1)      # [batch_size, 2]
                    next_quantiles = tf.gather_nd(next_quantiles, indices=indices)                  # [batch_size, quantiles]
                
                targets = tf.stop_gradient(rewards[:, None] + self.gamma**self.n_steps * next_quantiles * (1 - dones[:, None]))  # 加None可以直接添加一维
                
                # loss
                '''
                参看论文!!!!!!
                要计算 all pair of (theta_i, theta_j)
                dim=1是tau_hat维
                dim=2是target维
                '''
                huber_loss = self.huber_loss(targets[:, None, :], current_quantiles[:, :, None])  # [batch_size, quantiles, quantiles]
                
                bellman_errors = targets[:, None, :] - current_quantiles[:, :, None]              # [batch_size, quantiles, quantiles]
                # huber_loss = (  # Equation 9 of paper
                #     tf.cast(tf.abs(bellman_errors) <= 1, tf.float32) * 0.5 * bellman_errors ** 2 +
                #     tf.cast(tf.abs(bellman_errors) > 1, tf.float32) * 1 * (tf.abs(bellman_errors) - 0.5 * 1)
                #     )
                
                '''
                quantile midpoints. See Lemma 2 of paper
                '''
                tau_hat = (tf.range(self.quantiles, dtype=tf.float32) + 0.5) / self.quantiles
                '''
                Equation 10
                注意tau_hat是在前后各升一维，就是说tau_hat之后在tau_hat维上取平均是指dim=1
                '''
                quantile_huber_loss = tf.abs(tau_hat[None, :, None] - tf.cast(bellman_errors < 0, tf.float32)) * huber_loss
                '''
                average over target value dimension, sum over tau dimension
                '''
                loss = tf.reduce_sum(tf.reduce_mean(quantile_huber_loss, axis=2), axis=1)
                
                if self.prioritized:
                    loss = loss * tf.constant(weights)
                    self.memory.update_priorities(replay_buffer_indices, np.abs(tf.squeeze(loss).numpy()).tolist())
                
                loss = tf.reduce_mean(loss)
                
            grads = tape.gradient(loss, self.model.trainable_variables)
            # grads = [tf.clip_by_value(grad, -1, 1) for grad in grads]
        
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            self.losses.append(loss.numpy())
            
            # update target model
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())
            
            
            '''
            也可以使用soft update: traget_weight = (1 - 0.999) * model_weight + 0.999 * target_weight
            '''
            # params = self.model.get_weights()
            # target_params = self.target_model.get_weights()
            # for idx in range(len(params)):
            #     target_params[idx] = (1 - 0.999) * params[idx] + 0.999 * target_params[idx]
            # self.target_model.set_weights(target_params)
            
        
        def get_action(self, obs, eps=0.1, training=None):   # epsilon-greedy policy
            if np.random.random() >= eps or self.eval_mode:
                # print(s.dtype)
                obs = np.expand_dims(obs, 0)
                obs = tf.convert_to_tensor(obs, dtype=tf.float32)
                
                a = tf.argmax(tf.reduce_mean(self.model(obs, training=training), axis=-1), axis=-1).numpy()
                return int(a)
            else:
                return np.random.randint(0, self.action_dim)    
        
        
        def eval_(self, env, n_trajs):

            self.eval_mode = True

            for _ in range(n_trajs):
                episode_return = 0
                episode_length = 0
                obs = env.reset()
                for _ in range(10000):
                    a = self.get_action(obs, training=self.eval_mode)
                    obs, reward, done, info = env.step(a)
                    episode_return += reward
                    episode_length += 1
                    
                    if done:
                        self.rewards.append(episode_return)
                        self.episode_length.append(episode_length)

                        break
                        
            # print('eval {} trajs, mean return: {}'.format(n_trajs, np.mean(episode_returns)))
            self.eval_mode = False
            return np.mean(self.rewards[-n_trajs:]), np.max(self.rewards[-n_trajs:]), np.mean(self.episode_length[-n_trajs:]), np.max(self.episode_length[-n_trajs:])
        
        
        
        def n_steps_replay(self, transition):
            '''
            如果想要使用 n-steps TD learning,  在收集transition时就用这个函数，而不是用self.memory.add()
            模拟证明还是挺有用的，在LunarLander-v2上，其他DQN tricks都不加，只是用n-steps=5,就能训练得蛮好的，
            虽然前期的loss趋势很诡异，会先上升一段，然后开始慢慢下降
            '''
            _, _, _, obs_tpn, done = transition
            self.n_step_buffer.append(transition)
            
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            R = sum([self.n_step_buffer[i][2] * self.gamma**i for i in range(self.n_steps)])
            obs_t, action, _, _, _ = self.n_step_buffer.pop(0)
            
            self.memory.add((obs_t, action, R, obs_tpn, done))
        
        
        def save_w(self):
            self.model.save_weights('./model_weights.ckpt')
    
    
        def load_w(self):
            fname_model = './model_weights.ckpt'
            
            if os.path.isfile(fname_model):
                self.model.load_weights(fname_model)
                
                
        def save_replay(self):
            pickle.dump(self.memory, open('./exp_replay_agent.dump', 'wb'))
            
        def load_replay(self):
            fname = './exp_replay_agent.dump'
            if os.path.isfile(fname):
                self.memory = pickle.load(open(fname, 'rb'))
                
                
        def render(self, env):
            
            self.eval_mode = True
            obs = env.reset()
            for _ in range(10000):
                env.render()
                a = self.get_action(obs, training=self.eval_mode)
                obs, reward, done, info = env.step(a)
                if done:
                    break

            self.eval_mode = False