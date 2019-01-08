# Project 2 - Continuous Control

## Introduction
For this project, I used Version 2 environment (20 return version).  However, the learning algorithm isn't distributed and so doesn't fully take advantage of the 20x sample rate.  It does partially take advantage however, assuming that a large mini-batch run once is more efficient (on a GPU) than looping over a small mini batch 20 times. I suspect this increase is small however.  As will be seen toward the end, this provided a useful lesson.

## Scaling
Recall that Version 1 (V1) environment returns a single experience while Version 2 (V2) takes 20 actions and returns 20 experiences.

The buffer size was set to 1e6, which is arguably 20x what might be called for V1.  This has the impact that each of the 20 agents in V2 is sampling over the same time window that a single agent would be sampling over in V1 as the velocity through the buffer is 20x faster than V1.

The mini batch size was set to 20*128 or 2560.  I found other students' work, which looped 20 times over mini-batches of 128.  It seems to me that looping once over a large mini batch could be more efficient as the CPU has to at least attend to the loop.

One of the advantages I discovered of doing it this way, is that 1 time step in V2 is 20 in V1. For hyper parameter optimization, if you do a grid search, you only need to run a dozen time steps or less per grid. The criteria of >30 average over 100 episodes in V2 isn't really analogous to >30 over 100 episodes in V1 either. 

## Algorithm
An article on Soft Actor Critic popped into my feed as I started this project so my first attempt was with SAC. I spent several days working on it before switching back to DDPG. The code is included in the repository, but the code has not been cleaned up.  All of the files start with 'SAC' that relate to the soft actor critic attempt.

My project now employs vanilla DDPG taken straight from the lesson, but slightly adapted for the 20x return per time step, but not distributed.  I spent several full days trying to get a solution to converge on this as well. To get past this, in desperation, I compared my project to @tommytracey and finally discovered I had fixed what I though was a typo in the classroom provided model code which introduced a bug.  Otherwise, I would have spent that time attempting to do distributed version of DDPG or one of the other distributed learning algorithms.  Some of my hyper parameters are identical to his.

### Hyperparamers and other settings

hyperparameter|value
---|---
buffer_size     | 100,000|
batch_size      | 128*20|
gamma           | 0.99 t
tau            | 1e-3
LR_actor       | 1e-3
LR_critic      | 1e-3
weight_decay   | .00 
max_episodes   | 250
epsilon_decay  | .99995
fc1_units      | 400
fc2_units      | 300
sigma          | 0.1

### Pre-Fill
While chasing my bug and operating under the assumption that I just had the wrong hyper parameters, it occurred to me that pre populating the buffer with experiences from random actions might be an advantage.  It produced a definite increase in the initial returns in the buggy version.  As I write this, I running a random grid search on hyper parameters and the size of that pre-fill is one of the search parameters.  Right now it is unclear this actually helps.  My instinct is that it might in small numbers, say 2 or 3 x the mini batch size.  Beyond that, and I suspect it will dilute policy derived experiences.
In the final version, I took out that code.  It's affect was to increase the score for the first episode only and possibly decrease the second episode score.  Beyond the third episode, it had no discernable effect.

## Environment Description Improvement Opportunity
The environment description reports a reward of .1 for having the end of the arm in the target region.  I found this to be incorrect. The environment rewards are in the range 0-.0399... This makes sense as the maximum reward per episode appears to 40 (in my own experience and the training graph from the project instructions) and the environment has a fixed run time of 1000 time steps.  I assume the variable return represents the amount of time within the time step spent on target, but did not spent any time verifying that assumption.  As I was trying to figure out why my solution wouldn't converge, this discrepancy sent me down a rabbit hole. 

## Logging
After spending several days trying to achieve convergence, I was losing track of what I had previously tried.  To that end, I wrote some code to log hyper parameters and output. Early versions went to individual files. In certain cases, I manually copied output when the logging code was apparently broken. Later versions went to a single log file, continually appended. It is work in progress, but has proved useful - in fact very useful as will be shown in the next sub topic.

## Why does episode time increase 3 fold over course of training?
Why does training start out at about a minute per episode and end up at about 3 minutes per episode?  I struggle to imagine an explanation for that.

After writing the above sentence, I returned to this question and it occurred to me that the increase in episode duration increased linearly until 50 episodes which corresponds to when the buffer reaches capacity under V2 environment and 1e6 capacity.  Last night, I ran it at 1e5.  The graph closed when the program ended so all I have is my log, but it reached the criteria in 121 episodes (~1:20 episode duration), which was much faster than at 1e6. I'm now not sure how I convinced myself that 1e6 was near optimal.  I got the 1e6 value from another student's similar work (@tommytracey), but thought I had tried smaller values with less success. It may be that I did, but some other hyperparameter offset the gain.

I speculate that the 'named tupple' that contains the buffer information is pretty inefficient and if that large of a buffer were really necessary, it would be worth investigating a different structure, perhaps something that could be moved to the GPU.

## Results
Once I set good hyper parameters, it took about 30 episodes to reach 30 and completed in about 130. 

I briefly attempted crawler with this code, but was unsuccessful at getting it to learn.  It was peculiar.  It would end up sitting there quivering balanced on all four feet very close to each other.  

![training](https://github.com/shogan50/p2_continuous_control/blob/master/training.PNG?raw=true "training" )

## Future improvements
Obviously trying distributed algorithms would be a good place to start for improvements.  Before I discovered that the smaller buffer size works at least as well as the large one, I considered dynamically adjusting the minibatch size when the training reached it's asymptote. This wouldn't really improve training, but would meet the requirements of this class in less time.  However, it got me to thinking about what information is available in the returns and what might be done to take advantage of this. I don't fully grok SAC and while I believe I understand the meaning of the word entropy, I don't understand how it is used here in SAC or the other algorithms that include it in loss, but I suspect this is a method of using the information available in the returns.

