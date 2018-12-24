import numpy as np
import random
import math

import copy
from collections import namedtuple, deque

from SAC_model import Value_network, Soft_Q_network, Policy_network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SAC_agent():
    def __init__(self, state_size, action_size, buffer_size):
        self.replay_buffer = Replay_buffer(buffer_size=buffer_size)
        self.soft_q_net = Soft_Q_network(state_size,action_size,fc1_units=128,fc2_units=128).to(device)
        self.value_net = Value_network(state_size,action_size,fc1_units=128,fc2_units=128).to(device)
        self.target_value_net = Value_network(state_size,action_size,fc1_units=128,fc2_units=128).to(device)
        self.policy_net = Policy_network(state_size,action_size,fc1_units=128,fc2_units=128).to(device)

        self.value_optimizer = optim.Adam(self.value_net.parameters(),lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(),lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=policy_lr)

        for target_params, params in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_params.data.copy_(params.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.replay_buffer = Replay_buffer(buffer_size)


    def soft_q_update(self, batch_size, gamma=0.99, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0, soft_tau=1e-2):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)         # collect samples

        state = torch.FloatTensor(state).to(device)                                             # push to GPU
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = self.soft_q_net(state, action)                                       #
        expected_value = self.value_net.forward(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

class Replay_buffer:
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = []
        self.position = 0

    def push(self,state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.size

    def sample(self,batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


