# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym

env = gym.make("FetchPickAndPlace-v1")

state = env.reset()
a = env.step([0, 1, 0, 1])

print(a)
b = env.compute_reward(a[0]['achieved_goal'], a[0]['desired_goal'], a[3])
a = env.step([0, 0, 0, 1])

print(env.compute_reward(a[0]['achieved_goal'], a[0]['desired_goal'], a[3]))
print(a[0]['achieved_goal'])

n_states = env.observation_space
a_states = env.action_space

#print(help(env))