import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
import replay_buffer
import policy
from env import DroneEnv

torch.backends.cudnn.benchmark = True




class Workspace(object):
    def __init__(self):

        self.seed = 0
        self.device = "cuda"
        self.batch_size = 128
        self.buffer_capacity = 20000
        self.train_steps = 100000
        utils.set_seed_everywhere(self.seed)

        self.device = torch.device(self.device)
        self.env = DroneEnv()

        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = policy.Agent(obs_shape, action_shape, action_range,
                                         self.device, discount=0.99, init_temperature=0.1,
                                         lr=2.5e-4, actor_update_frequency=2, critic_tau=0.01,
                                         critic_target_update_frequency=2, batch_size=self.batch_size,
                                         log_std_bounds=[-10, 2], hidden_dim=1024, hidden_depth=2,
                                         feature_dim=8)

        self.buffer = replay_buffer.ReplayBuffer(obs_shape, action_shape,
                                                     capacity=self.buffer_capacity,
                                                     device=self.device)

        self.step = 0
        self.eval_episodes = 20

    def run(self):
        episode, episode_reward, episode_step, done, done_no_max = 0, 0, 1, False, False
        start_time = time.time()
        seed_steps = 1000
        num_train_iters = 1
        obs, others = self.env.reset()
        loss = []
        episode_rewards = []
        sr = []
        aer = []

        while self.step < self.train_steps:

            if done:
                episode_rewards.append(episode_reward)
                obs, others = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                print(self.step)

            if self.step < seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.select_action(obs, others, sample=True)


            next_obs, next_others, reward, done, _ = self.env.step(action)

            if self.step >= seed_steps:
                for i in range(1):
                    a_loss = self.agent.update(self.buffer, self.step)
                    loss.append(a_loss)

            if (episode_step + 1) == self.env._max_episode_steps:
                done = 1.0

            done = float(done)
            done_no_max = 0 if (episode_step + 1) == self.env._max_episode_steps else done
            episode_reward += reward


            self.buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max, others, next_others)
            obs = next_obs
            others = next_others
            episode_step += 1
            self.step += 1

            if self.step % 10000 == 0:

                self.agent.save()

            if self.step % 10000 == 0:

                self.env.client.simPause(True)
                np.savetxt("loss.txt", np.array(loss))
                np.savetxt("episode_reward.txt", np.array(episode_rewards))
                self.env.client.simPause(False)

            if self.step % 2000 == 0:
                x, y = self.evaluate()
                sr.append(x)
                aer.append(y)
                np.savetxt("eval_success_rate.txt", np.array(sr))
                np.savetxt("eval_episode_reward.txt", np.array(aer))
                done = True

    def evaluate(self):
        average_episode_reward = 0
        success_rate = 0
        self.env.index_eval = 0
        self.env.goal_eval = [0.0, 0.0]
        for episode in range(self.eval_episodes):
            obs, others = self.env.reset_eval()
            done = False
            episode_reward = 0
            episode_step = 0
            arrived = False
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.select_action(obs, others, sample=False)
                obs, others, reward, done, arrived = self.env.step_eval(action)
                if (episode_step + 1) == self.env._max_episode_steps:
                    done = 1.0

                episode_reward += reward
                episode_step += 1
            if arrived:
                success_rate += 1
            average_episode_reward += episode_reward
        success_rate /= self.eval_episodes
        average_episode_reward /= self.eval_episodes

        return success_rate, average_episode_reward


def main():
    workspace = Workspace()

    workspace.run()


if __name__ == '__main__':
    main()
