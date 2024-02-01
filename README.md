# Reinforcement Learning Algorithms

该GitHub仓库收录了我整理的关于强化学习的算法，目前包括以下一些算法：

- 无需梯度的方法：Cross-Entropy method
- 非深度学习的经典RL算法：SARSA和Q-learning for Q-table（针对离散状态空间和离散动作空间）
- 深度学习和强化学习结合的算法
- - Value-based
  - - Deep Q Network (DQN)
    - Double DQN
    - Dueling DQN
    - Prioritized DQN
    - Categorical DQN
    - Quantile DQN
  - Policy-based
  - - REINFORCE
    - Policy Gradient with Baseline
  - Actor-Critic
  - - Adavantage Actor Critic (A2C)

所有深度强化学习的算法都包含4个基本大块组成：

- Config：用来设定各种模型的参数
- Network：用来定义各种算法的网络结构
- Agent：用来与环境进行交互的方法
- Training Loop：用来训练和评估的过程

Value-based方法还会涉及到一个Experience Replay Memory。

最基础的DQN和REINFORCE方法我是在jupyter notebook里完整地写了整个逻辑的代码，剩下的方法就是在jupyter notebook里调用了我分别写在 `Config.py`，`networks.py`，`agents.py`和 `replay_memories.py`里的类，但是逻辑都是类似的，在jupyter notebook里就写的是Training Loop，然后在 `gym`里的 `CartPole-v0`和 `LunarLander-v2`上跑了一下写的算法（只有部分写了在Lunarlander上跑的结果，因为跑的时间比较长）。

所有的方法我争取用Pytorch和TensorFlow2都实现一遍，目前已经整理好了Pytorch的部分（也不能算完全整理好吧，因为Categorical DQN还没有在 `LunarLander-v2`上跑通关，而A2C在 `CartPole-v0`上跑了好久才通关，`CartPole-v0`算是RL中的‘Hello world’级别的游戏，这个没跑通的话，说明写得有些问题...），TensorFlow版的写完了DQN的Variants和REINFORCE。

我还会持续向这里面添加我学习的RL的算法，另外如果也想了解一下具体算法的原理，也欢迎移步[我的博客主页](https://leyuanheart.github.io/Blogs/)，上面有关于相应RL算法的介绍。

Python以及主要的package的版本：

- python=3.6.8
- tensorflow==2.1.0
- tensorflow-probability==0.9
- torch==1.6.0
- torchvision== 0.7.0
- gym==0.17.2
- numpy==1.18.5
- matplotlib==3.2.2

---

新补充的 `PPO`，`DDPG`，`TD3`和 `SAC`只有Pytorch版本的，代码是单独在各自的文件夹中。

`Dyna-Q`算法在 `Dyna-Q.ipynb`中。

Python以及package有更新：

- python=3.7.16
- torch==1.13.0
- gymnasium==0.26.3（最主要的变化是这个）
- numpy==1.21.5
- matplotlib==3.5.2
