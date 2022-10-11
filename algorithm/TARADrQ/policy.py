import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils


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
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])



class Actor(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, feature_dim=8):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.log_std_bounds = log_std_bounds

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2 * action_shape[0])
        )

        self.lstm = nn.LSTMCell(8, 8)
        self.ta = nn.Linear(2 * 8, 2)
        self.outputs = dict()
        self.apply(utils.weight_init)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs, others, detach_encoder=False):

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
        h_list = []

        for x in xs:
            ht, ct = self.lstm(x, (ht, ct))
            h_list.append(ht)

        total_ht = h_list[0]
        for i in range(1, len(h_list)):
            total_ht = torch.cat((total_ht, h_list[1]), 1)
        beta_t = self.relu(self.ta(total_ht))
        beta_t = self.softmax(beta_t)
        out = torch.zeros(obs.size()[0], 8).to("cuda")
        for i in range(len(h_list)):
            out = out + h_list[i] * beta_t[:, i].reshape(obs.size()[0], 1)

        obs = out

        obs = torch.cat([obs, others], dim=1)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()


        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth, feature_dim=8):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.Q1 = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.lstm = nn.LSTMCell(8, 8)
        self.ta = nn.Linear(2 * 8, 2)
        self.outputs = dict()
        self.apply(utils.weight_init)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, obs, action, others, detach_encoder=False):
        assert obs.size(0) == action.size(0)

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
        h_list = []

        for x in xs:
            ht, ct = self.lstm(x, (ht, ct))
            h_list.append(ht)

        total_ht = h_list[0]
        for i in range(1, len(h_list)):
            total_ht = torch.cat((total_ht, h_list[1]), 1)
        beta_t = self.relu(self.ta(total_ht))
        beta_t = self.softmax(beta_t)
        out = torch.zeros(obs.size()[0], 8).to("cuda")
        for i in range(len(h_list)):
            out = out + h_list[i] * beta_t[:, i].reshape(obs.size()[0], 1)

        obs = out
        obs = torch.cat([obs, others], dim=1)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class Agent(object):

    def __init__(self, obs_shape, action_shape, action_range, device,
                 discount, init_temperature, lr, actor_update_frequency,
                 critic_tau, critic_target_update_frequency, batch_size,
                 log_std_bounds, hidden_dim, hidden_depth, feature_dim, target_num):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = Actor(obs_shape, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, feature_dim).to(device)

        self.critic = Critic(obs_shape, action_shape, hidden_dim, hidden_depth, feature_dim).to(device)
        self.critic_target = []
        for i in range(target_num):
            self.critic_target.append(Critic(obs_shape, action_shape, hidden_dim, hidden_depth, feature_dim).to(device))
            self.critic_target[i].load_state_dict(self.critic.state_dict())

        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_shape[0]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        for i in range(target_num):
            self.critic_target[i].train()

        self.target_index = 0
        self.target_num = target_num

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, others, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        others = torch.FloatTensor(others).to(self.device)
        dist = self.actor(obs, others)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, others, next_others):
        with torch.no_grad():
            dist = self.actor(next_obs, next_others)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q = 0.0
            for i in range(self.target_num):
                target_Q1, target_Q2 = self.critic_target[i](next_obs, next_action, next_others)
                target_V = torch.min(target_Q1,
                                     target_Q2) - self.alpha.detach() * log_prob
                target_Q += reward + (not_done * self.discount * target_V)
            target_Q = target_Q / self.target_num
            dist_aug = self.actor(next_obs_aug, next_others)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            target_Q_aug = 0.0
            for i in range(self.target_num):
                target_Q1, target_Q2 = self.critic_target[i](next_obs_aug, next_action_aug, next_others)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                target_Q_aug += reward + (not_done * self.discount * target_V)
            target_Q_aug = target_Q_aug / self.target_num
            target_Q = (target_Q + target_Q_aug) / 2

        current_Q1, current_Q2 = self.critic(obs, action, others)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action, others)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach()

    def update_actor_and_alpha(self, obs, others):
        dist = self.actor(obs, others, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, others, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()


        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.log_alpha.grad.data.clamp_(-1, 1)
        self.log_alpha_optimizer.step()


    def update(self, replay_buffer, step):

        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug,\
        others, next_others = replay_buffer.sample(self.batch_size)
        critic_loss = self.update_critic(obs, obs_aug, action, reward, next_obs,
                                              next_obs_aug, not_done, others, next_others)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, others)

        if step % self.critic_target_update_frequency == 0:
            self.target_index = (self.target_index + 1) % self.target_num
            utils.soft_update_params(self.critic, self.critic_target[self.target_index],
                                     self.critic_tau)
        return critic_loss

    def save(self, save_i):
        torch.save(self.actor.state_dict(), "actor_" + str(save_i))
        torch.save(self.critic.state_dict(), "critic_" + str(save_i))

    def load(self, save_i):
        self.actor.load_state_dict(torch.load("actor_" + str(save_i)))
