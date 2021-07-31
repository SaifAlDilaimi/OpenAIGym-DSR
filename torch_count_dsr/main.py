n_episodes = 300
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.985
BETA = 0.025

import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
from gym import wrappers
from gym_minigrid.wrappers import *


def preprocess(img: np.array) -> np.array:
    # resize image
    output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = output.flatten()
    output = output.astype('float32')
    output = np.array([output])
    return output

env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field

state_size = 3136
action_size = 3

agent = Agent(state_size, action_size, seed = 0)
eps = eps_start 
done = False
reward_per_episode = []

for i_episode in range(n_episodes):
    episodic_reward = 0
    state0 = env.reset()
    state0 = preprocess(state0)
    
    for t in range(max_t):
        action, r_int = agent.parallel_act(state0, eps)
        state1, reward, done, _ = env.step(action)
        state1 = preprocess(state1)
        r = reward + BETA * np.array(r_int)
        agent.step(state1, action, r, state1, done)
        state0 = state1
        episodic_reward += reward
        if done:
            break

    reward_per_episode.append(episodic_reward)
    eps = max(eps_end, eps_decay*eps) # decrease epsilon

    if i_episode % 10 == 0 or i_episode <= 5:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(episodic_reward)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

# Save figure
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(np.arange(len(reward_per_episode)), reward_per_episode)
plt.ylabel('Score')
plt.xlabel('Episode Number')
plt.savefig('Results.png')
# Save data
df = pd.DataFrame(reward_per_episode)
df.to_csv('score.csv', index=False)
print("figure + data saved")
    