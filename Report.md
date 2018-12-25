# Project 2 - Continuous Control

## Introduction
While I used Version 2 environment (20 return version), the learning algorithm isn't distributed and so doesn't fully take advantage of the 20x sample rate.  It does partially take advantage however, assuming that a large mini-batch run once is more efficient (on a GPU) than looping over a small mini batch 20 times.

## Scaling
Recall that Version 1 (V1) environment returns a single experience while Version 2 (V2) takes 20 actions and returns 20 experiences.

The buffer size was set to 1 million, which is arguably 20x what would be called for V1.  This scaling means each of the 20 agents in V2 is sampling over the same time window that a single agent would be sampling over in V1.

The mini batch size was set to 20*128 or 2560.  I found other students' work, which looped 20 (or 200) times over mini-batches of 128.  It seems to me that looping once over a large mini batch would be more efficient.

One of the advantages I discovered of doing it this way, is that 1 time step in V2 is 20 in V1. For hyper parameter optimization, if you do a grid search, you only need to run a dozen time steps or less per grid. The criteria of >30 average over 100 episodes in V2 isn't really analogous to >30 over 100 episodes in V1. 

## Algorithm
My project employs vanilla DDPG, optimized for the 20x return per time step, but not distributed.  I spent several full days trying to get a solution to converge.  I had fixed what I though was a typo in the classroom provided model code which introduced a bug.  Otherwise, I would have spent that time attempting to do distributed version of DDPG or one of the other distributed learning algorithms.
### Pre-Fill
While chasing my bug and operating under the assumption that I just had the wrong hyper parameters, it occurred to me that pre populating the buffer with experiences from random actions might be an advantage.  It produced a definite increase in the initial returns in the buggy version.  As I write this, I running a random grid search on hyper parameters and the size of that pre-fill is one of the search parameters.  Right now it is unclear this actually helps.  My instinct is that it might in small numbers, say 2 or 3 x the mini batch size.  Beyond that, and I suspect it will dilute policy derived experiences.

## Environment Description Improvement Opportunity
The environment description reports a reward of .1 for having the end of the arm in the target region.  I found this to be incorrect. The environment rewards are in the range 0-.0399... This makes sense as the maximum reward per episode appears to 40 (in my own experience and the training graph from the project instructions) and the environment has a fixed run time of 1000 time steps.  I assume the variable return represents the amount of time within the time step spent on target, but did not spent any time verifying that assumption.  As I was trying to figure out why my solution wouldn't converge, this discrepancy sent me down a rabbit hole. 

## Logging
After spending several days trying to achieve convergence, I was losing track of what I had previously tried.  To that end, I wrote some code to log hyper parameters and output. Early versions went to individual files.  Later versions went to a single log file, continually appended. It is work in progress, but has proved useful.



