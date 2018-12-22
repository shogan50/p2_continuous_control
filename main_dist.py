from unityagents import UnityEnvironment
import numpy as np
import os
from matplotlib import pyplot as plot

from CC_agent_dist import Agent
import time
import torch
import torch.multiprocessing as mp
from model import Critic, Actor


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

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

episodes = 250
max_t = 1000
episode_t = []
learn = False
scores_hist = []
explore = 100.
epsilon = 1
epsilon_min = .05

for episode in range(episodes):
    start_t = time.time()
    env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
    scores = np.zeros(num_agents)
    actions = []
    epsilon -= 1 / explore
    epsilon = max(epsilon,epsilon_min)
    states = env_info.vector_observations  # get the current state (for each agent)
    for t in range(max_t):
        actions = agent.act(states, epsilon)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations          # get next state (for each agent)
        rewards = env_info.rewards                          # get reward (for each agent)
        dones = env_info.local_done                         # see if episode finished
        scores += env_info.rewards                          # update the score (for each agent)
        learn = True if t%1 == 0 else False
        agent.step(states, actions, rewards, next_states, dones, learn, 1)
        states = next_states
        if np.any(dones):  # exit loop if episode finished
            break
    episode_t = time.time()-start_t
    scores_hist.append(scores)
    time_remain = np.mean(episode_t)*(episodes-episode)
    if episode % 1 == 0:
        print('Score (ave. over agents) for ep. {}: {:04f} \tT: {:00.0f}:{:02.1f}(m:s)\tEst remain: {:02.0f}:{:02.0f}(h:m)'.format(episode, np.mean(scores),episode_t//60,episode_t%60, time_remain//3600,time_remain%3600/60))
    if len(scores_hist)>100 and np.mean(scores_hist[:-100]) >= 30:
        print('Finished in {} episodes'.format(episode+1))
plt = plot.plot(scores_hist)
plt.show()

#
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

