import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import utils as utils

import augmentations as augmentations
from sac import SacAgent

from utils import (
    compute_attribution,
    compute_attribution_mask,
)
import random


class AttributionDecoder(nn.Module):
    def __init__(self, action_shape, emb_dim=100) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim+action_shape, out_features=14112)
        self.conv1 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, padding=1)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        x = self.proj(x).view(-1, 32, 21, 21)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class AttributionPredictor(nn.Module):
    def __init__(self, action_shape, encoder, emb_dim=100):
        super().__init__()
        self.encoder = encoder
        self.decoder = AttributionDecoder(action_shape, encoder.out_dim)
        self.features_decoder = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, emb_dim)
        )

    def forward(self, x,action):
        x = self.encoder(x)
        return self.decoder(x, action)


class SGQNAgent(SacAgent):
    def __init__(self, obs_shape, action_shape, device, args):
        super().__init__(obs_shape, action_shape, device, args)

        self.attribution_predictor = AttributionPredictor(action_shape[0], self.critic.encoder).cuda()
        self.quantile = 0.95
        self.auxiliary_update_freq = args.auxiliary_update_freq

        self.aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(),
            lr=args.aux_lr,
            betas=(args.aux_beta, 0.999),
        )

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, WB_LOG=None):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        masked_obs = obs * mask
        # mask 为 0 的像素点随机采样
        masked_obs[mask < 1] = random.uniform(obs.view(-1).min(), obs.view(-1).max())
        masked_Q1, masked_Q2 = self.critic(masked_obs, action)
        critic_loss += 0.5 * (F.mse_loss(current_Q1, masked_Q1) + F.mse_loss(current_Q2, masked_Q2))
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_aux(self, obs, action, obs_grad, mask, step=None, L=None):
        # mask = compute_attribution_mask(obs_grad, self.quantile)
        # s_prime = augmentations.attribution_augmentation(obs.clone(), mask.float()) #  这个干嘛的？

        s_tilde = augmentations.random_overlay(obs.clone())
        # s_tilde = augmentations.identity(obs.clone())

        self.aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde, action, mask)
        aux_loss.backward()
        self.aux_optimizer.step()

        if L is not None:
            L.log("train/aux_loss", aux_loss, step)

    def compute_attribution_loss(self, obs, action, mask):
        mask = mask.float()
        attrib = self.attribution_predictor(obs.detach(), action.detach())
        aux_loss = F.binary_cross_entropy_with_logits(attrib, mask.detach())
        return attrib, aux_loss

    def update(self, replay_buffer, L, step, WB_LOG):

        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_drq()  # obs next_obs pos 都使用随机 shift
        self.update_critic(obs, action, reward, next_obs, not_done, L, step, WB_LOG)
        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, quantile=self.quantile)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, WB_LOG)

        if step % self.critic_target_update_freq == 0:
            # 这里包括 共享 target 编码器、两个计算 Q 值target网络、自己定义的辅助任务的下一状态转移模型 target 网络
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)

        if step & self.auxiliary_update_freq == 0:
            # 重点：辅助任务，传入 random shift 后的当前观测、下一观测、以及当前观测的copy
            self.update_aux(obs, action, obs_grad, mask, step, L)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f"{model_dir}/actor_{step}.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/critic_{step}.pt")

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f"{model_dir}/actor_{step}.pt", map_location='cuda'))
        self.critic.load_state_dict(torch.load(f"{model_dir}/critic_{step}.pt", map_location='cuda'))
