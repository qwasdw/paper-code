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
import sub_replay_buffer
import master_replay_buffer
import sub_policy
import master_policy
from video import VideoRecorder
from env import DroneEnv

torch.backends.cudnn.benchmark = True




class Workspace(object):
    def __init__(self):

        self.seed = 0
        self.device = "cuda"
        self.batch_size = 128
        self.master_buffer_capacity = 20000
        self.sub_buffer_capacity = 20000
        self.train_steps = 100000
        utils.set_seed_everywhere(self.seed)

        self.device = torch.device(self.device)
        self.env = DroneEnv()

        self.num_subpolicy = 3
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.master_agent = master_policy.MasterAgent(obs_shape, self.device, discount=0.99,
                                                      lr=2.5e-4, batch_size=self.batch_size,
                                                      num_subpolicy=self.num_subpolicy,hidden_dim=1024,
                                                      hidden_depth=2,feature_dim=8)
        self.master_buffer = master_replay_buffer.ReplayBuffer(obs_shape,
                                                               capacity=self.master_buffer_capacity,
                                                               device=self.device)
        self.sub_agent = []
        self.sub_buffer = []
        for i in range(self.num_subpolicy):
            self.sub_agent.append(sub_policy.SubAgent(obs_shape, action_shape, action_range,
                                             self.device, discount=0.99, init_temperature=0.1,
                                             lr=2.5e-4, actor_update_frequency=2, critic_tau=0.01,
                                             critic_target_update_frequency=2, batch_size=self.batch_size,
                                             log_std_bounds=[-10, 2], hidden_dim=1024, hidden_depth=2,
                                             feature_dim=8, target_num=4))
            self.sub_buffer.append(sub_replay_buffer.ReplayBuffer(obs_shape, action_shape,
                                                         capacity=self.sub_buffer_capacity,
                                                         image_pad=4,
                                                         device=self.device))

        self.step = 0
        self.master_step = 0
        self.eval_episodes = 40
        self.sub_steps = 5


    def run(self):
        episode, episode_reward, episode_step, done, done_no_max = 0, 0, 1, False, False
        start_time = time.time()
        sub_steps = self.sub_steps
        seed_steps = 1000
        num_train_iters = 1
        obs, others = self.env.reset()
        M_loss = []
        S_loss = [[] for _ in range(self.num_subpolicy)]
        M_reward = []
        sr = []
        aer = []


        while self.step < self.train_steps:

            if done:
                print(self.step)

                master_loss = self.master_agent.update(self.master_buffer)
                M_loss.append(master_loss)

                obs, others = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1


            master_action = self.master_agent.select_action(obs)
            master_obs = obs
            master_next_obs = None
            master_reward = 0.0
            for i in range(sub_steps):

                if self.step < seed_steps:
                    sub_action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.sub_agent[master_action]):
                        sub_action = self.sub_agent[master_action].select_action(obs, others, sample=True)


                next_obs, next_others, reward, done, _ = self.env.step(sub_action)

                if self.step >= seed_steps:
                    for i in range(self.num_subpolicy):
                        sub_loss = self.sub_agent[i].update(self.sub_buffer[i], self.step)
                        S_loss[i].append(sub_loss)

                if (episode_step + 1) == self.env._max_episode_steps:
                    done = 1.0
                    reward = -10.0

                done = float(done)
                done_no_max = 0 if (episode_step + 1) == self.env._max_episode_steps else done
                episode_reward += reward

                master_reward += reward

                self.sub_buffer[master_action].add(obs, sub_action, reward, next_obs, done,
                                       done_no_max, others, next_others)
                obs = next_obs
                master_next_obs = next_obs
                others = next_others
                episode_step += 1
                self.step += 1

                if self.step % 10000 == 0:
                    self.master_agent.save()
                    for i in range(self.num_subpolicy):
                        self.sub_agent[i].save(i)

                if self.step % 10000 == 0:
                    self.env.client.simPause(True)
                    for i in range(self.num_subpolicy):
                        np.savetxt("sub_loss_" + str(i) + ".txt", np.array(S_loss[i]))
                    np.savetxt("master_loss.txt", np.array(M_loss))
                    np.savetxt("master_reward.txt", np.array(M_reward))
                    self.env.client.simPause(False)

                if self.step % 5000 == 0:
                    x, y = self.evaluate()
                    sr.append(x)
                    aer.append(y)
                    np.savetxt("eval_success_rate.txt", np.array(sr))
                    np.savetxt("eval_episode_reward.txt", np.array(aer))
                    done = True

                if done:
                    break
            self.master_step += 1
            self.master_buffer.add(master_obs, master_action, master_reward,
                                   master_next_obs, done, done_no_max)
            M_reward.append(master_reward)



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
                master_action = self.master_agent.select_action(obs)
                for i in range(self.sub_steps):
                    with utils.eval_mode(self.sub_agent[master_action]):
                        action = self.sub_agent[master_action].select_action(obs, others, sample=False)
                    obs, others, reward, done, arrived = self.env.step_eval(action)
                    if (episode_step + 1) == self.env._max_episode_steps:
                        done = 1.0
                    if done:
                        break

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
