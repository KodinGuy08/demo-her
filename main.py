import random
import gym
from Agent import Agent as DDPG
import numpy as np

import threading

from utils import plot_learning_curve
from pathlib import Path
is_env_on = Path("~/env")

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

env = gym.make("FetchPickAndPlace-v1")

#env.unwrapped.viewer.window.on_key_press = key_press
#env.unwrapped.viewer.window.on_key_release = key_release

state = env.reset()
n_states = state['observation'].shape[0]
n_goal = state['desired_goal'].shape[0]

n_actions = len(env.action_space.high)
action_bound = env.action_space.high[0]

gamma = 0.99
lr_actor =0.0001
lr_critic = 0.001

tau = 0.001

dense_layers = [256, 256]

DEMO_REFILL = False

import sys

args = sys.argv

agent = DDPG(alpha=lr_actor, beta=lr_critic, input_dims=[n_states], goal_dims=[n_goal], tau=tau, env=env,
              batch_size=1536, layer_size=dense_layers, n_actions=n_actions,
              chkpt_dir=args[1], Datagen=True, load=(args[2]=="1"))

import time

start = time.time()

PERIOD_OF_TIME = 7200*4 # 120 min

print("### Start Learning Mode ###")

ep_run = False
#Done no longer counts

success_rates = []
reward_q = []

ep_t = []

print("Looping...", env._max_episode_steps)
for ep in range(40000):
    ep_run = True

    state_ = env.reset()
    goal = state_['desired_goal']
    state = state_['observation']

    step_count = 0

    states = []
    new_states = []
    a_goals = []

    actions = []

    rewards = []
    dones = []
    infos = []

    score = []

    ep_time = time.time()

    while ep_run:
        if is_env_on.exists() or ep > 20000 or True:
            pass

        ng = agent.subtract_array(state_['achieved_goal'].copy(), goal.copy())

        action, a_c = agent.choose_action(state, ng), 1

        actions.append(action)

        for a in range(a_c):
            new_state, reward, done, info = env.step(action)

        if args[3] == "1":
            env.render()

        state_ = new_state.copy()

        score.append(info['is_success'])

        states.append(state.copy())
        a_goals.append(new_state['achieved_goal'].copy())
        rewards.append(reward.copy())
        infos.append(info.copy())

        dones.append(done)

        new_state = new_state['observation'].copy()
        new_states.append(new_state.copy())
        state = new_state.copy()

        if done and step_count > env._max_episode_steps:
            break

        step_count = step_count + 1

    for step in range(step_count):
        r = rewards[step]

        g = a_goals[step].copy()
        ng = agent.subtract_array(g, goal.copy())

        agent.remember(states[step], actions[step], r, new_states[step], int(dones[step]),
                        ng)
        for i in range(0, 4):
            if 4 > len(a_goals):
                break
            g = a_goals[-(i+1)].copy()
            s = states[-(i+1)].copy()
            a = actions[-(i+1)].copy()
            ns = new_states[-(i+1)].copy()
            d = dones[-(i+1)]

            info = {"is_success":1.0}
            goal_ = g.copy()
            r1 = env.compute_reward(g, g, info)

            agent.remember(s, a, r1, ns, 1,
                           agent.subtract_array(g, goal_))
    if ep % 1000 == 0 and ep > 0 and DEMO_REFILL:
        agent.datagen()

        for i in range(100):
            agent.learn()

    

    for i in range(step_count):
        if not arg[3] == "1":
            agent.learn()

    if ep % 25 == 0 and ep > 0:
        agent.save_models()

    if ep % 1000 == 0 and False:
        eval = agent.eval()
        #plot_learning_curve(28, eval, "out.png")

    success_count = 0

    for s in score:
        if s > 0.92:
            success_count = success_count + 1
    success_rates.append(success_count/len(score))
    reward_q.append(sum(rewards))

    ep_t.append(ep_time - time.time())
    print("Current Episode: ", ep, " Success Rate:", success_rates[-1], "Time Avg:", np.mean(ep_t[-100:]))

    print()
    plot_learning_curve([i for i in range(ep+1)], success_rates, "out_success_rate.png")
    #plot_learning_curve([i for i in range(ep+1)], success_rates, "out_fake_test.png", False)
    plot_learning_curve([i for i in range(ep+1)], reward_q, "out_reward.png", False)
    #plot_learning_curve([i for i in range(ep+1)], reward_q, "out_reward_.png")

