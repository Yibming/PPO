import gym
import torch
import numpy as np

import os.path
import time

from ppo_agent import PPOAgent

scenario="Pendulum-v1"
env = gym.make(scenario)

#Directory for saving models
current_path = os.path.dirname(os.path.realpath(__file__) )
model_path = current_path + "/models/"
timestamp = time.strftime("%Y%m%dT%H%M%S")

NUM_EPISODE = 3000
NUM_STEP = 200
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE = 25
UPDATE_INTERVAL=50  # 第UI步开始学习


agent=PPOAgent(STATE_DIM, ACTION_DIM,BATCH_SIZE)    #TODO
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_reward = -2000

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    done = False
    episode_reward = 0
    
    for step_i in range(NUM_STEP):
        action,value = agent.get_action(state)  #TODO
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        done = True if (step_i + 1) == NUM_STEP else False
        agent.replay_buffer.add_memo(state, action, reward, value, done)
        state = next_state

        if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1) == NUM_STEP:
            agent.update()  # TODO

    if episode_reward  >= -100 and episode_reward >best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model_path + f"ppo_actor_{timestamp}.pth")
        print(f"Best reward: {best_reward}")


    REWARD_BUFFER[episode_i] = episode_reward
    print(f"EPISODE:{episode_i}, Reward: {round(episode_reward,2)}")

env.close

