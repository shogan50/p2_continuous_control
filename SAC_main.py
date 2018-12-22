from unityagents import UnityEnvironment
import numpy as np
import os
from matplotlib import pyplot as plot
from SAC_agent import SAC_agent
from utils import plot


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

replay_size = 100000

agent = SAC_agent(state_size=state_size, action_size=action_size, buffer_size=replay_size)

max_episodes = 250
max_t = 1000
episode_t = []
scores_hist = []
batch_size = 128
step_idx = 0
rewards = []

for episode_idx in range(max_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    episode_rewards = np.zeros(state_size)

    done = False
    while not done:
        actions = []
        for st_idx in range(len(states)):
            actions.append(agent.policy_net.get_action(states[0]))
        next_states, rewards, dones, _ = env.step(actions)

        agent.replay_buffer.push(states,actions,rewards,next_states,dones)
        if len(agent.replay_buffer) > batch_size:
            agent.soft_q_update(batch_size)

        states = next_states
        episode_rewards += rewards

        step_idx +=1

        if step_idx % 1000 == 0:
            plot(step_idx,np.mean(rewards,1))

        if np.any(dones):
            break

    rewards.append(episode_rewards)










