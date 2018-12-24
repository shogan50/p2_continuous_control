from unityagents import UnityEnvironment
import numpy as np
import os
from matplotlib import pyplot as plt
from SAC_agent import SAC_agent
from utils import plot
import time

import torch



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

states = env_info.vector_observations
state_size = states.shape[1]
print('Size of each state:', state_size)

max_episodes = 500
episode_t = []
scores_hist = []
batch_size = 256
step_idx = 0
rewards = []
replay_size = 1000000
soft_q_repeats = 10

agent = SAC_agent(state_size=state_size, action_size=action_size, buffer_size=replay_size)

for episode_idx in range(max_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    episode_rewards = np.zeros(num_agents)
    start_t = time.time()
    done = False
    while not done:
        actions = []

        for st_idx in range(len(states)):
            actions.append(agent.policy_net.get_action(states[st_idx]))

        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations          # get next state (for each agent)
        rewards = env_info.rewards                          # get reward (for each agent)
        dones = env_info.local_done                         # see if episode finished
        episode_rewards += rewards                          # update the score (for each agent)
        for st_idx in range(len(states)):
            agent.replay_buffer.push(states[st_idx], actions[st_idx], rewards[st_idx], next_states[st_idx], dones[st_idx]) #TODO send all,  for now we send just the first
        if len(agent.replay_buffer) > batch_size:
            for _ in range(soft_q_repeats):
                agent.soft_q_update(batch_size)

        states = next_states

        step_idx +=1

        # if step_idx % 1000 == 0:
        #     plot(step_idx,np.mean(rewards))

        if np.any(dones):
            break

    rewards.append(np.mean(episode_rewards))
    episode_t = time.time() - start_t
    time_remain = np.mean(episode_t)*(max_episodes-episode_idx)
    if episode_idx % 1 == 0:
        print('Score ([min, mean, max] over agents) for ep. {}: [{:0.2f},{:02.f},{:01.f}] \tT: {:00.0f}:{:02.0f}(m:s)\tEst remain: {:00.0f}:{:02.0f}(h:m)'\
              .format(episode_idx, np.min(episode_rewards),np.mean(episode_rewards),np.max(episode_rewards),episode_t//60,episode_t%60, time_remain//3600,time_remain%3600/60))
    if len(scores_hist)>100 and np.mean(scores_hist[:-100]) >= 30:
        print('Met project requirement in {} episodes'.format(episode_idx+1))
        torch.save(agent.soft_q_net.state_dict(),'soft_q.pth')
        torch.save(agent.value_net.state_dict(),'value.pth')
        torch.save(agent.target_value_net.state_dict(),'value.pth')
        torch.save(agent.policy_net.state_dict(),'policy.pth')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(rewards) + 1), rewards)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

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













