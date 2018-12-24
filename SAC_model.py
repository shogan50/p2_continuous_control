import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Value_network(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=128,init_w=3e-3):
        super(Value_network,self).__init__()
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.bc0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc3.weight.data.uniform_(-init_w,init_w)
        self.fc3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state):
        x = self.bc0(state)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Soft_Q_network(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=128,init_w=3e-3):
        super(Soft_Q_network,self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn0 = nn.BatchNorm1d(state_size+action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc3.weight.data.uniform_(-init_w,init_w)
        self.fc3.bias.data.uniform_(-init_w,init_w)

    def forward(self,state,action):
        x = torch.cat([state,action],1)
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Policy_network(nn.Module):
    def __init__(self,state_size, action_size, fc1_units = 128, fc2_units=128, init_w=3e-3, log_std_min=-20, log_std_max = 2):
        super(Policy_network,self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.bn0 = nn.BatchNorm1d(state_size)


        self.fc_mean = nn.Linear(fc2_units,action_size)
        self.fc_mean.weight.data.uniform_(-init_w,init_w)
        self.fc_mean.bias.data.uniform_(-init_w,init_w)

        self.fc_log_std = nn.Linear(fc2_units,action_size)
        self.fc_log_std.weight.data.uniform_(-init_w,init_w)
        self.fc_log_std.bias.data.uniform_(-init_w,init_w)

    def forward(self, state):
        # x = self.bn0(state)
        x = F.relu(self.fc1(state))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std,self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std