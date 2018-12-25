from unityagents import UnityEnvironment
import numpy as np
import os
import platform
from matplotlib import pyplot as plt


from DDPG_agent import Agent
import time
import datetime
import torch
from utils import log_me
import torch.multiprocessing as mp
from DDPG_Model import Critic, Actor


dir = os.getcwd()
dir = dir + os.sep + "Reacher_Windows_x86_64"
env = UnityEnvironment(file_name=dir + os.sep + "Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def DDPG(env, kwargs):

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0, kwargs=kwargs)

    max_episodes = kwargs['max_episodes']
    epsilon_decay = kwargs['epsilon_decay']
    learn_every = kwargs['learn_every']
    learn_repeat = kwargs['learn_repeat']
    pre_fill_qty = kwargs['pre_fill_qty']

    max_t = 1000
    episode_t = []
    learn = False
    scores_hist = []
    epsilon = 1
    epsilon_min = .05

    log = log_me('ddpg_log.log')
    log.add_line('********************************************************************************************')
    log.add_line('time :' + str(datetime.datetime.now()))
    log.add_line('main file: ' + os.path.basename(__file__))
    log.add_line('platform = ' + platform.system())
    log.add_line('device = ' + str(agent.device))
    log.save_lines()
    log.add_kwargs(kwargs)

    log.log('filling buffer with data from random actions')


    while len(agent.memory) < agent.buffer_size and len(agent.memory) < pre_fill_qty:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        done = False
        while not done:
            actions = np.random.randn(num_agents, action_size)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            # scores += env_info.rewards  # update the score (for each agent)
            for idx in range(len(rewards)):
                agent.memory.add(states[idx], actions[idx],rewards[idx], next_states[idx], dones[idx])
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        log.log("buffer volume: ", len(agent.memory), " of ", pre_fill_qty)
    log.log('Training Started')


    for episode in range(max_episodes):
        start_t = time.time()
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        agent.reset()
        scores = np.zeros(num_agents)
        states = env_info.vector_observations  # get the current state (for each agent)
        for t in range(max_t):
            actions = agent.act(states, epsilon)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            scores += env_info.rewards                          # update the score (for each agent)
            learn = True if t % learn_every == 0 else False
            agent.step(states, actions, rewards, next_states, dones, learn, learn_repeat)
            states = next_states
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)
            if np.any(dones):  # exit loop if episode finished
                break
        episode_t = time.time()-start_t
        scores_hist.append(scores)
        time_remain = np.mean(episode_t)*(max_episodes - episode)
        if episode % 1 == 0:
            log_line = 'Score ([min, mean, max] over agents) for ep. {}: [{:0.2f},{:0.2f},{:0.1f}] \tT: {:00.0f}:{:02.0f}(m:s)\tEst remain: {:00.0f}:{:02.0f}(h:m)' \
                .format(episode, np.min(scores), np.mean(scores), np.max(scores),
                        episode_t // 60, episode_t % 60, time_remain // 3600, time_remain % 3600 / 60)
            log.log(log_line)


        if len(scores_hist) > 100 and np.mean(scores_hist[:-100]) >= 30:
            print('Met project requirement in {} episodes'.format(episode + 1))
            torch.save(agent.soft_q_net.state_dict(), 'soft_q.pth')
            torch.save(agent.value_net.state_dict(), 'value.pth')
            torch.save(agent.target_value_net.state_dict(), 'value.pth')
            torch.save(agent.policy_net.state_dict(), 'policy.pth')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(1, len(rewards) + 1), rewards)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()
    return scores_hist

ddpg_args = {"buffer_size": int(1e6), # replay buffer size
             "batch_size": 128*2, # minibatch size
             "gamma" : 0.99,  # discount factor
             "tau" : 1e-3,  # for soft update of target parameters
             "LR_actor" : 1e-3,  # learning rate of the actor
             "LR_critic" : 1e-3,  # learning rate of the critic
             "weight_decay" : .00 , # L2 weight decay
             "max_episodes": 15,
             "epsilon_decay" : .99995,
             "learn_every" : 1,
             "learn_repeat" : 1,
             "pre_fill_qty" : 128*10,
             "fc1_units" : 400,
             "fc2_units" : 300,
             "sigma" : 0.2
              }

done = False
import random


while not done:
    DDPG(env, kwargs=ddpg_args)
    ddpg_args['batch_size'] = random.randint(128*2,128*40)
    ddpg_args['pre_fill_qty'] = random.randint(0,100000)
    ddpg_args['sigma'] = random.random()*.2

# import multiprocessing
#
# env1 = UnityEnvironment(file_name=dir + os.sep + "Reacher.exe")
# env2 = UnityEnvironment(file_name=dir + os.sep + "Reacher.exe")

