from unityagents import UnityEnvironment
import numpy as np
import os
from config import Config
from CC_agent import Agent
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

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
config = Config()
config.num_workers = num_agents
config.task_fn = lambda: Task(game, num_envs=config.num_workers,
                              log_dir=get_default_log_dir(a2c_continuous.__name__),
                              single_process=True)
config.eval_env = Task(game)
config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
config.network_fn = lambda: GaussianActorCriticNet(
    config.state_dim, config.action_dim,
    actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
config.discount = 0.99
config.use_gae = True
config.gae_tau = 1.0
config.entropy_weight = 0.01
config.rollout_length = 5
config.gradient_clip = 5
config.max_steps = int(2e7)
config.logger = get_logger(tag=a2c_continuous.__name__)
run_steps(A2CAgent(config))

#
# agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
#
# episodes = 500
# max_t = 1000
# episode_t = []
#
#
# for episode in range(episodes):
#     start_t = time.time()
#     env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
#     scores = np.zeros(num_agents)
#     actions = []
#     for t in range(max_t):
#         states = env_info.vector_observations               # get the current state (for each agent)
#
#         for agent_i in range(num_agents):
#             actions.append(agent.act(states[agent_i]))
#         env_info = env.step(actions)[brain_name]
#         next_states = env_info.vector_observations          # get next state (for each agent)
#         rewards = env_info.rewards                          # get reward (for each agent)
#         dones = env_info.local_done                         # see if episode finished
#         scores += env_info.rewards                          # update the score (for each agent)
#         agent_i = np.random.randint(0, num_agents-1)
#         agent.step(states[agent_i], actions[agent_i], rewards[agent_i], next_states[agent_i], dones[agent_i])
#         states = next_states
#         if np.any(dones):  # exit loop if episode finished
#             break
#     episode_t = time.time()-start_t
#     time_remain = np.mean(episode_t)*(episodes-episode)
#     if episode % 1 == 0:
#         print('Total score (averaged over agents) for episode {}: {} \test time remaining: {}'.format(episode, np.mean(scores), time_remain))
#
#




#
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

