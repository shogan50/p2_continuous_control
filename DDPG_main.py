from unityagents import UnityEnvironment
import numpy as np
import os
import platform
from matplotlib import pyplot as plt

from CC_agent_dist import Agent
import time
import datetime
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
learn_every = 1
learn_repeat = 2

cwd = os.getcwd()
f_name = 'run log '
id = 0
while os.path.exists(cwd + os.sep + f_name + str(id) + '.log'):
    id+=1
log_path = cwd + os.sep + f_name + str(id) + '.log'
f = open(log_path,mode='w')
f.write('time {} {}:{}'.format(datetime.datetime.day, datetime.datetime.hour, datetime.datetime.min))
f.write('main file: ' + os.path.basename(__file__))
f.write('platform = ' + platform.system())
f.write('\nepisodes = '+ str(episodes))
f.write('\nbuffer size = ' + str(agent.buffer_size))
f.write('\ngamma = ' + str(agent.gamma))
f.write('\ntau = ' + str(agent.tau))
f.write('\nlearning rate actor = ' + str(agent.LR_actor))
f.write('\nlearning rate critic = ' + str(agent.LR_critic))
f.write('\ndevice = ' + str(agent.device))
f.write('\nlearn every = ' + str(learn_every))
f.write('\nlearn repeat x times = ' + str(learn_repeat))
f.write('\n(noise) sigma = ' + str(agent.sigma))

f.close()

print('filling buffer with data from random actions')
while len(agent.memory) < agent.buffer_size:
    env_info = env.reset(train_mode=True)[brain_name]
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
    print("buffer volume: ", len(agent.memory), "of ",agent.buffer_size)
print('Training Started')
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
        learn = True if t%learn_every == 0 else False
        agent.step(states, actions, rewards, next_states, dones, learn, learn_repeat)
        states = next_states
        if np.any(dones):  # exit loop if episode finished
            break
    episode_t = time.time()-start_t
    scores_hist.append(scores)
    time_remain = np.mean(episode_t)*(episodes-episode)
    if episode % 1 == 0:
        log_line = 'Score ([min, mean, max] over agents) for ep. {}: [{:0.2f},{:0.2f},{:0.1f}] \tT: {:00.0f}:{:02.0f}(m:s)\tEst remain: {:00.0f}:{:02.0f}(h:m)' \
            .format(episode, np.min(scores), np.mean(scores), np.max(scores),
                    episode_t // 60, episode_t % 60, time_remain // 3600, time_remain % 3600 / 60)
        print(log_line)
        f = open(log_path,mode='a')
        f.write('\n' + log_line)
        f.close()

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

