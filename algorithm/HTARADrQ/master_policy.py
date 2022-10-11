import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra


class Encoder(nn.Module):

    def __init__(self, obs_shape, feature_dim=8):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 8
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2)
        ])


        self.outputs = dict()

    def forward_conv(self, obs):
        obs = torch.unsqueeze(obs, dim=1)
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = nn.functional.adaptive_avg_pool2d(h, (1,1))
        out = torch.squeeze(out, 3)
        out = torch.squeeze(out, 2)


        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

class Master(nn.Module):

    def __init__(self, obs_shape, feature_dim, hidden_dim, hidden_depth, num_subpolicy):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.Q = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_subpolicy)
        )
        self.lstm = nn.LSTMCell(8, 8)


        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs0 = self.encoder(obs[:, 0, :, :], detach=detach_encoder)
        obs1 = self.encoder(obs[:, 1, :, :], detach=detach_encoder)

        obs2 = self.encoder(obs[:, 2, :, :], detach=detach_encoder)
        obs3 = self.encoder(obs[:, 3, :, :], detach=detach_encoder)

        xs = []
        xs.append(obs0)
        xs.append(obs1)
        xs.append(obs2)
        xs.append(obs3)
        ht = torch.zeros(obs.size()[0], 8).to("cuda")
        ct = torch.zeros(obs.size()[0], 8).to("cuda")
        for x in xs:
            ht, ct = self.lstm(x, (ht, ct))

        obs = ht

        q = self.Q(obs)

        return q

class MasterAgent(object):

    def __init__(self, obs_shape, device, discount,
                 lr, batch_size, num_subpolicy,
                 hidden_dim, hidden_depth, feature_dim):
        self.device = device
        self.discount = discount
        self.batch_size = batch_size

        self.master = Master(obs_shape, feature_dim, hidden_dim, hidden_depth, num_subpolicy).to(device)
        self.target = Master(obs_shape, feature_dim, hidden_dim, hidden_depth, num_subpolicy).to(device)

        # optimizers
        self.master_optimizer = torch.optim.Adam(self.master.parameters(), lr=lr)

        self.epsilon = 0.5
        self.num_subpolicy = num_subpolicy
        self.update_count = 0

        self.train()

    def train(self, training=True):
        self.training = training
        self.master.train(training)

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        value = self.master(obs)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= self.epsilon:  # epslion greedy
            action = np.random.choice(range(self.num_subpolicy), 1).item()
        action = np.array(action)
        return action

    def update(self, replay_buffer):
        total_loss = 0.0
        if replay_buffer.idx > self.batch_size:
            data_count = 30
            for _ in range(data_count):
                obs, action, reward, next_obs, not_done = replay_buffer.sample(self.batch_size)
                reward = (reward - reward.mean()) / (reward.std() + 1e-7)
                with torch.no_grad():
                    target_v = reward + self.discount * self.target(next_obs).max(1)[0].unsqueeze(1) * not_done

                a = target_v
                b = self.master(obs).gather(1, action)
                loss = F.mse_loss(a, b)
                self.master_optimizer.zero_grad()
                loss.backward()
                self.master_optimizer.step()
                self.update_count += 1
                if self.update_count % 50 == 0:
                    self.target.load_state_dict(self.master.state_dict())
                total_loss += loss.detach()
            total_loss /= data_count
            self.epsilon = 1 - (1 - self.epsilon) * 0.99


        return total_loss

    def save(self):
        torch.save(self.master.state_dict(), "master")

    def load(self):
        self.master.load_state_dict(torch.load("master"))
