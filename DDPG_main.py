from unityagents import UnityEnvironment
import numpy as np
import os
import platform
import matplotlib.pyplot as plt


from DDPG_agent import Agent
import time
import datetime
import torch
from utils import log_me, Logger
import torch.multiprocessing as mp
from DDPG_Model import Critic, Actor
import sys

sys.stdout = Logger('ddpg_log.log')
print('************************************************************************************')
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

def DDPG(env, kwargs):

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0, kwargs=kwargs)

    max_episodes = kwargs['max_episodes']
    epsilon_decay = kwargs['epsilon_decay']

    #TODO: refactor kwargs.  It is really a dictionary and won't work as ** variable.

    max_t = 1000
    scores_hist = []
    epsilon = 1
    epsilon_min = .05


    print('time :' + str(datetime.datetime.now()))
    print('main file: ' + os.path.basename(__file__))
    print('platform = ' + platform.system())
    print('device = ' + str(agent.device))
    print(kwargs)

    line, = plt.plot([],[])
    axes = plt.gca()
    plt.ion()
    plt.xlabel = 'Episode'
    plt.ylabel = 'Mean score'
    #TODO: these labels aren't showing up on figure.

    for episode in range(max_episodes):
        start_t = time.time()                                   #capture the episode start time  TODO: it would be nice to capture start of train time and report clock duration of solve.
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        agent.reset()
        scores = np.zeros(num_agents)                           #reset the episode scores tally
        states = env_info.vector_observations  # get the current state (for each agent)
        for t in range(max_t):
            actions = agent.act(states, epsilon)                #   get actions (20 in this case)
            env_info = env.step(actions)[brain_name]            # get return from environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            scores += env_info.rewards                          # update the score (for each agent)
            agent.step(states, actions, rewards, next_states, dones) #save SARSD to buffer and learn from buffered SARSD
            states = next_states                                # prep for next pass
            epsilon *= epsilon_decay                            # explore a wee bit less
            epsilon = max(epsilon, epsilon_min)                 # always explore a little
            if np.any(dones):  # exit loop if episode finished
                break
        episode_t = time.time()-start_t                         # wow that took a long time.
        scores_hist.append(np.mean(scores))                     # keep track of score history
        time_remain = np.mean(episode_t)*(max_episodes - episode)# are we there yet?
        if episode % 1 == 0:
            print('Score ([min, mean, max] over agents) for ep. {}: [{:0.2f},{:0.2f},{:0.1f}] \tT: {:00.0f}:{:02.0f}(m:s)\tEst remain: {:00.0f}:{:02.0f}(h:m)' \
                .format(episode, np.min(scores), np.mean(scores), np.max(scores),
                        episode_t // 60, episode_t % 60, time_remain // 3600, time_remain % 3600 / 60))

            line.set_xdata(np.arange(0,len(scores_hist)))
            line.set_ydata(scores_hist)
            axes.set_xlim(0,len(scores_hist))
            axes.set_ylim(0,np.max(scores_hist)*1.05)
            plt.draw()
            plt.pause(.1)

        if episode % 10 == 0 and episode > 20:          #Lets occasionally save the weights just in case
            torch.save(agent.critic_target.state_dict(), 'critic_target.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_local.pth')
            torch.save(agent.actor_local.state_dict(), 'actor_target.pth')
            torch.save(agent.actor_target.state_dict(), 'actor_local.pth')

        if len(scores_hist) > 100 and np.min(scores_hist[-100:]) >= 30:  # yippee!
            print('Met project requirement in {} episodes'.format(episode + 1))
    return scores_hist

# the args are done this way as a convenient method of recording
# the hyper paramaters associated with each run
ddpg_args = {"buffer_size": int(1e5), # replay buffer size
             "batch_size": 128*20, # minibatch size
             "gamma" : 0.99,  # discount factor
             "tau" : 1e-3,  # for soft update of target parameters
             "LR_actor" : 1e-3,  # learning rate of the actor
             "LR_critic" : 1e-3,  # learning rate of the critic
             "weight_decay" : .00 , # L2 weight decay
             "max_episodes": 250,
             "epsilon_decay" : .99995,
             "fc1_units" : 400,
             "fc2_units" : 300,
             "sigma" : 0.1
              }

done = False
DDPG(env, kwargs=ddpg_args)

plt.show()

## The code below was used for an unattended grid search.  At one point is was a loop with the hyperparameters selected randomly.
# ddpg_args["max_episodes"] = 50
# ddpg_args['fc1_units'] = 128
# ddpg_args['fc2_units'] = 128
# DDPG(env, kwargs=ddpg_args)
#
# ddpg_args['fc1_units'] = 400
# ddpg_args['fc2_units'] = 300
# ddpg_args['LR_actor'] = (1e-3)*2
# DDPG(env, kwargs=ddpg_args)
#
# ddpg_args['LR_actor'] = (1e-3)/2
# DDPG(env, kwargs=ddpg_args)
#
# ddpg_args['LR_actor'] = 1e-3
# ddpg_args['LR_critic'] = (1e-3)*2
# DDPG(env, kwargs=ddpg_args)
#
# ddpg_args['LR_critic'] = (1e-3)/2
# DDPG(env, kwargs=ddpg_args)


