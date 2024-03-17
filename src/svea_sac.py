import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
import wandb
import augmentations as augmentations
import math
import random
from sac import SacAgent

from utils import (
    compute_attribution,
    compute_attribution_mask,
)

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SVEAAgent(SacAgent):
    def __init__(self, obs_shape, action_shape, device, args):  

        super().__init__(obs_shape, action_shape, device, args)

        self.action_repeat = args.action_repeat

        # 各种 loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()


    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, WB_LOG=None):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)


        critic_loss = 0.5 * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        )
       
        obs_aug = augmentations.random_overlay(obs.clone())
        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        critic_loss += 0.5 * (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update(self, replay_buffer, L, step, WB_LOG):

        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_drq()  # 随机 shift
        
       
        self.update_critic(obs, action, reward, next_obs, not_done, L, step, WB_LOG)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, WB_LOG)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f"{model_dir}/actor_{step}.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/critic_{step}.pt")

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f"{model_dir}/actor_{step}.pt", map_location='cuda'))
        self.critic.load_state_dict(torch.load(f"{model_dir}/critic_{step}.pt", map_location='cuda'))
