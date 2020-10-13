#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:20:23 2020

@author: didi
"""

import numpy as np
import pandas as pd
import pickle
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.list_physical_devices(device_type='GPU')



class MLP_tensorflow(keras.Model):
    def __init__(self, action_dim):
        super(MLP_tensorflow, self).__init__()
        self.action_dim = action_dim
        
        self.fc1 = layers.Dense(64)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(64)
        self.fc4 = layers.Dense(self.action_dim)
        
    def call(self, obs, training=None):   # obs is set to be tensor before inputing the model
        # x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.relu(self.fc1(obs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x



class DuelingNetwork_tensorflow(keras.Model):
    def __init__(self, action_dim):
        super(DuelingNetwork_tensorflow, self).__init__()
        self.action_dim = action_dim
        
        self.fc1 = layers.Dense(64)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(64)
        
        # vlaue head
        self.value = layers.Dense(1)
        # adv head
        self.adv = layers.Dense(self.action_dim)
        
        
    def call(self, obs, training=None):   # obs is set to be tensor before inputing the model
        # x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.relu(self.fc1(obs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        
        value = self.value(x)
        adv = self.adv(x)
        
        adv = adv - tf.reduce_mean(adv,keepdims=True)
        
        return value + adv






class QuantileNetwork_tensorflow(keras.Model):
    def __init__(self, action_dim, quantiles=51):
        super(QuantileNetwork_tensorflow, self).__init__()
        self.action_dim = action_dim
        self.quantiles = quantiles
        
        self.fc1 = layers.Dense(64)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(64)
        self.fc4 = layers.Dense(self.action_dim*self.quantiles)
        
    def call(self, obs, training=None):   # obs is set to be tensor before inputing the model
        # x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.relu(self.fc1(obs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = self.fc4(x)
        
        return tf.reshape(x, (-1, self.action_dim, self.quantiles))






class CONV_tensorflow(keras.Model):
    def __init__(self, action_dim):
        super(CONV_tensorflow, self).__init__()
        
        self.action_dim = action_dim
        
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(self.action_dim)
        
    def call(self, obs, training=None):    # obs must be tensor
        # x = tf.cast(obs, dtype=tf.float32)
        x = tf.nn.relu(self.conv1(obs))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))  # keras.activations.relu(self.conv3(x))
        x = self.flatten(x)
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)

        return x